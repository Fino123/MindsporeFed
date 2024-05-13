
package com.example.mindsporefederatedlearning.autoencoder;

import android.util.Log;

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
 * 类名：AutoencoderDataset
 * 功能：定义用户画像模型的数据结构
 * @author Administrator
 */
public class AutoencoderDataset extends DataSet {
    /** 日志 */
    private static final Logger LOGGER = Logger.getLogger(AutoencoderDataset.class.toString());
    /** 模型的运行模式 */
    private final RunType runType;
    /** 一组 UserFeature 类型的用户数据，是 AutoEncoder 模型的第一个输入数据 */
    private List<UserFeature> features;
    /** 一组用户的真实标签，是 AutoEncoder 模型的第二个输入数据 */
    private List<List<Integer>> targetLabels;

    /**
     * 构造函数
     * @param batchSize: 一批数据的大小
     * @param runType: 模型模式
     */
    public AutoencoderDataset(RunType runType, int batchSize){
        this.runType = runType;
        this.batchSize = batchSize;
        this.features = new ArrayList<>();
    }

    /**
     * 方法名：getTargetLabels()
     * 功能：获取真实标签
     * @return targetLabels：真实标签
     */
    public List<List<Integer>> getTargetLabels() {
        return targetLabels;
    }


    /**
     * 方法名：readTxtFile()
     * 功能：读取数据文件
     * @param file：文件名
     * @return allLines：数据文本
     */
    private static List<String> readTxtFile(String file){
        if(file == null){
            LOGGER.severe("file cannot be empty");
            return new ArrayList<>();
        }
        Path path = Paths.get(file);
        List<String> allLines = new ArrayList<>();
        try {
            allLines = Files.readAllLines(path, StandardCharsets.UTF_8);
        } catch (IOException e) {
            LOGGER.severe("read txt file failed, please check txt file path");
        }
        return allLines;
    }


    /**
     * 方法名：convertData()
     * 功能：从文件中读取数据，并装载进 features 中
     * @param dataFile：数据文件
     * @param labelFile：标签文件
     * @return status：是否成功
     */
    private Status convertData(String dataFile, String labelFile){
        if (dataFile == null || labelFile == null) {
            LOGGER.severe("convert data failed, dataFile and labelFile cannot be empty");
            return Status.NULLPTR;
        }
        // read file
        List<String> dataLines = readTxtFile(dataFile);
        List<String> labelLines = readTxtFile(labelFile);
        // set up data
        targetLabels = new ArrayList<>(labelLines.size());
        for (int i=0; i<dataLines.size(); i++) {
            // parsing data
            String dataStr = dataLines.get(i);
            String[] dataTokens = dataStr.split(" ");
            List<Float> data = new ArrayList<>(dataTokens.length);
            for (String dataToken : dataTokens){
                data.add(Float.valueOf(dataToken));
            }
            // parsing labels
            String labelStr = labelLines.get(i);
            String[] labelTokens = labelStr.split(" ");
            List<Integer> labels = new ArrayList<>(labelTokens.length);
            for (String labelToken : labelTokens){
                labels.add(Integer.valueOf(labelToken));
            }
            features.add(new UserFeature(data, labels));
        }
        sampleSize = features.size();
        Log.d("convert train data", "finished");
        return Status.SUCCESS;
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
        }

        // 数据缓冲区
        ByteBuffer input = inputsBuffer.get(0);
        input.order(ByteOrder.nativeOrder());

        // 标签缓冲区
        ByteBuffer label = inputsBuffer.get(1);
        label.order(ByteOrder.nativeOrder());

        // 将第 batchIdx 个批次的共 batchSize 个数据和标签输送到对应缓冲区
        for (int i = 0; i < batchSize; i++) {
            UserFeature feature = features.get(batchIdx * batchSize + i);
            for (int j = 0; j < feature.data.size(); j++) {
                input.putFloat(feature.data.get(j));
                label.putFloat(feature.data.get(j));
            }
            // this label is used for kmeans
            targetLabels.add(feature.labels);
        }
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
        if(batchSize <= 0){
            LOGGER.severe("batch size should bigger than 0");
            return;
        }
        Log.d("padding, batchsize", String.valueOf(batchSize));
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
        return convertData(dataFile, labelFile);
    }

}

