package com.example.mindsporefederatedlearning.mlp;

import com.example.mindsporefederatedlearning.common.CommonParameter;

import com.mindspore.flclient.model.DataSet;
import com.mindspore.flclient.model.RunType;
import com.mindspore.flclient.model.Status;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * 类名：MlpDataset
 * 功能：定义APP预测模型的数据结构
 * @author Administrator
 */
public class MlpDataset extends DataSet {
    /** 日志 */
    private static final Logger LOGGER = Logger.getLogger(MlpDataset.class.toString());
    /** 模型的运行模式 */
    private final RunType runType;
    /** 一批数据的大小 */
    private final int batchSize;
    /** 用户数据 */
    private final List<AppFeature> features;
    /** 数据掩码 */
    private List<List<Integer>> targetMasks;
    /** 标签 */
    private List<List<Integer>> targetLabels;

    /**
     * 方法名：getTargetMasks()
     * 功能：获取数据掩码
     * @return targetMasks：数据掩码
     */
    public List<List<Integer>> getTargetMasks() {
        return targetMasks;
    }

    /**
     * 方法名：getTargetLabels()
     * 功能：获取真实标签
     * @return targetLabels: 真实标签
     */
    public List<List<Integer>> getTargetLabels() {
        return targetLabels;
    }

    /**
     * 构造函数
     * @param runType：模式运行模式
     * @param batchsize：一批数据的大小
     */
    public MlpDataset(RunType runType, int batchsize){
        this.runType = runType;
        this.batchSize = batchsize;
        this.features = new ArrayList<>();
    }

    /**
     * 方法名：fillInputBuffer()
     * 功能：将 features 按照批次输送到模型的输入缓冲区中
     * @param inputsBuffer：模型的输入缓冲区
     * @param batchIdx：批次编号
     */
    @Override
    public void fillInputBuffer(List<ByteBuffer> inputsBuffer, int batchIdx) {
        for (ByteBuffer inputBuffer : inputsBuffer) {
            inputBuffer.clear();
            inputBuffer.order(ByteOrder.nativeOrder());
        }

        // 数据缓冲区
        int[] inputsSep = {1,1,1,1,5,5,5,5};

        for (int i = 0; i < batchSize; i++) {
            AppFeature feature = features.get(batchIdx * batchSize + i);
            int start = 0;
            for (int j = 0; j < inputsSep.length; j++) {
                ByteBuffer inputBuffer = inputsBuffer.get(j);
                for (int k = 0; k < inputsSep[j]; k++) {
                    inputBuffer.putInt(feature.data.get(start++));
                }
            }
        }

        // 掩码缓冲区
        ByteBuffer mask = inputsBuffer.get(8);

        int maskSize = CommonParameter.maxMask * batchSize;
        int maskIndex = 0;
        List<Integer> maskIds = new ArrayList<>();
        for (int i=0;i<batchSize;i++){
            AppFeature feature = features.get(batchIdx * batchSize + i);
            for (int j = 0; j < feature.mask.size(); j++) {
                int maskAdding = (feature.mask.get(j)+ (i * CommonParameter.CLASS_NUM));
                maskIds.add(maskAdding);
                mask.putInt(maskAdding);
                maskIndex++;
                if (maskIndex>=maskSize) {
                    break;
                }
            }
            if (maskIndex>=maskSize) {
                break;
            }
        }
        int padding = maskSize - maskIndex;
        for (int i = 0; i < padding; i++) {
            mask.putInt(maskIds.get(i%maskIndex));
        }

        for (int i = 0; i < batchSize; i++) {
            AppFeature feature = features.get(batchIdx * batchSize + i);
            targetMasks.add(feature.mask);
        }

        // 标签缓冲区
        ByteBuffer label = inputsBuffer.get(9);
        for (int i = 0; i < batchSize; i++) {
            AppFeature feature = features.get(batchIdx * batchSize + i);
            for (int j = 0; j < CommonParameter.CLASS_NUM; j++){
                if (feature.label.contains(j)){
                    label.putFloat(1.0f);
                }
                else {
                    label.putFloat(0.0f);
                }
            }
            targetLabels.add(feature.label);
        }

    }


    /**
     * 方法名：readTxtFile()
     * 功能：读取数据文件
     * @param file：文件名
     * @return allLines：数据文本
     */
    private static List<String> readTxtFile(String file) {
        if (file == null) {
            LOGGER.severe("file cannot be empty");
            return new ArrayList<>();
        }
        Path path = Paths.get(file);
        List<String> allLines = new ArrayList<>();
        try {
            allLines = Files.readAllLines(path, StandardCharsets.UTF_8);
        } catch (IOException e) {
            LOGGER.severe("read txt file failed,please check txt file path");
        }
        return allLines;
    }

    /**
     * 方法名：convertTrainData()
     * 功能：从文件中读取训练数据，并装载进 features 中
     * @param dataFile：数据文件
     * @param labelFile：标签文件
     * @param maskFile：掩码文件
     * @return status：是否成功
     */
    private Status convertTrainData(String dataFile, String labelFile, String maskFile) {
        if (dataFile == null || labelFile == null || maskFile == null) {
            LOGGER.severe("dataset init failed,trainFile,idsFile,vocabFile cannot be empty");
            return Status.NULLPTR;
        }
        // read train file
        List<String> dataLines = readTxtFile(dataFile);
        List<String> labelLines = readTxtFile(labelFile);
        List<String> maskLines = readTxtFile(maskFile);
        for (int i=0; i<dataLines.size(); i++) {
            // 读取数据文件
            String dataStr = dataLines.get(i);
            String[] dataTokens = dataStr.split(" ");
            List<Integer> data = new ArrayList<>(dataTokens.length);
            for (String dataToken : dataTokens){
                data.add(Float.valueOf(dataToken).intValue());
            }
            // 读取标签文件
            String labelStr = labelLines.get(i);
            if (labelStr.isEmpty()) {
                continue;
            }
            String[] labelTokens = labelStr.split(",");
            List<Integer> label = new ArrayList<>(labelTokens.length);
            for (String labelToken : labelTokens){
                label.add(Integer.valueOf(labelToken));
            }
            // 读取掩码文件
            String maskStr = maskLines.get(i);
            String[] maskTokens = maskStr.split(",");
            List<Integer> mask = new ArrayList<>(maskTokens.length);
            for (String maskToken: maskTokens){
                mask.add(Integer.valueOf(maskToken));
            }
            // 将数据、标签、掩码封装进 features
            features.add(new AppFeature(data, label, mask));
        }
        sampleSize = features.size();
        targetLabels = new ArrayList<>(sampleSize);
        targetMasks = new ArrayList<>(sampleSize);
        return Status.SUCCESS;
    }

    @Override
    public void shuffle() {

    }

    /**
     * 方法名：padding()
     * 功能：对于数量未达到 baschsize 的一批数据，填充随机数据
     */
    @Override
    public void padding() {
        if (batchSize <= 0) {
            LOGGER.severe("batch size should bigger than 0");
            return;
        }
        LOGGER.info("before pad samples size:" + features.size());
        int curSize = features.size();
        int modSize = curSize - curSize / batchSize * batchSize;
        int padSize = modSize != 0 ? batchSize - modSize : 0;
        for (int i = 0; i < padSize; i++) {
            int idx = (int) (Math.random() * curSize);
            features.add(features.get(idx));
        }
        batchNum = features.size() / batchSize;
        LOGGER.info("after pad samples size:" + features.size());
        LOGGER.info("after pad batch num:" + batchNum);
    }

    /**
     * 方法名：dataPreprocess()
     * 功能：封装数据加载过程
     * @param files：数据文件和标签文件列表
     * @return status：是否加载成功
     */
    @Override
    public Status dataPreprocess(List<String> files) {
        String dataFile = files.get(0);
        String labelFile = files.get(1);
        String maskFile = files.get(2);
        return convertTrainData(dataFile, labelFile, maskFile);
    }
}
