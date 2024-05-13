package com.example.mindsporefederatedlearning.mlp;

import android.view.accessibility.AccessibilityNodeInfo;

import com.example.mindsporefederatedlearning.albert.AlbertDataSet;
import com.example.mindsporefederatedlearning.albert.CustomTokenizer;
import com.example.mindsporefederatedlearning.albert.Feature;
import com.example.mindsporefederatedlearning.common.CommonParameter;
import com.mindspore.MSTensor;
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
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.logging.Logger;

public class MLPDataset extends DataSet {
    private static final Logger LOGGER = Logger.getLogger(MLPDataset.class.toString());
    private static final int HIST = 3;
    private static final int MAX_APP = 5;
    private static final int INPUT_DIM = 24;

    private final RunType runType;
    private final int batchSize;
    private final List<AppFeature> features;

    public List<List<Integer>> getTargetMasks() {
        return targetMasks;
    }

    private List<List<Integer>> targetMasks;

    public List<List<Integer>> getTargetLabels() {
        return targetLabels;
    }


    private List<List<Integer>> targetLabels;

    public MLPDataset(RunType runType, int batch_size){
        this.runType = runType;
        this.batchSize = batch_size;
        this.features = new ArrayList<>();
    }

    @Override
    public void fillInputBuffer(List<ByteBuffer> inputsBuffer, int batchIdx) {
        for (ByteBuffer inputBuffer : inputsBuffer) {
            inputBuffer.clear();
            inputBuffer.order(ByteOrder.nativeOrder());
        }

//        ByteBuffer batchUser = inputsBuffer.get(0);
//        ByteBuffer batchCountry = inputsBuffer.get(1);
//        ByteBuffer batchDevice = inputsBuffer.get(2);
//        ByteBuffer batchNum = inputsBuffer.get(3);
//        ByteBuffer batchHistIds = inputsBuffer.get(4);
//        ByteBuffer batchHistClasses = inputsBuffer.get(5);
//        ByteBuffer batchHistCounts = inputsBuffer.get(6);
//        ByteBuffer batchHistTimes = inputsBuffer.get(7);
        int[] inputs_sep = {1,1,1,1,5,5,5,5};
//        for (int i = 0; i < inputs_sep.length; i++) {
//            ByteBuffer inputBuffer = inputsBuffer.get(i);
//            inputBuffer = ByteBuffer.allocateDirect(batchSize*inputs_sep[i]*8);
//            inputBuffer.order(ByteOrder.nativeOrder());
//            inputsBuffer.set(i, inputBuffer);
//        }

        for (int i = 0; i < batchSize; i++) {
            AppFeature feature = features.get(batchIdx * batchSize + i);
            int start = 0;
            for (int j = 0; j < inputs_sep.length; j++) {
                ByteBuffer inputBuffer = inputsBuffer.get(j);
                for (int k = 0; k < inputs_sep[j]; k++) {
                    inputBuffer.putInt(feature.data.get(start++));
                }
            }
        }

        ByteBuffer mask = inputsBuffer.get(8);

        int mask_size = CommonParameter.maxMask * batchSize;
        int mask_index = 0;
        List<Integer> mask_ids = new ArrayList<>();
        for (int i=0;i<batchSize;i++){
            AppFeature feature = features.get(batchIdx * batchSize + i);
            for (int j = 0; j < feature.mask.size(); j++) {
                int mask_adding = (feature.mask.get(j)+ (i * CommonParameter.CLASS_NUM));
                mask_ids.add(mask_adding);
                mask.putInt(mask_adding);
                mask_index++;
                if (mask_index>=mask_size)
                    break;
            }
            if (mask_index>=mask_size)
                break;
        }
        int padding = mask_size - mask_index;
        for (int i = 0; i < padding; i++) {
            mask.putInt(mask_ids.get(i%mask_index));
        }
//        int mask_num = 0;
//        for (int i = 0; i < batchSize; i++) {
//            AppFeature feature = features.get(batchIdx * batchSize + i);
//            feature.mask.forEach(v->mask.putInt(Math.toIntExact(v)));
//            mask_num += feature.mask.size();
//        }
//        int padding_mask = MAX_MASK * batchSize - mask_num;
//        for (int i = 0; i < batchSize; i++) {
//            AppFeature feature = features.get(batchIdx * batchSize + i);
//            for (int j = 0; j < feature.mask.size(); j++) {
//                padding_mask--;
//                if (padding_mask<0){
//                    break;
//                }else {
//                    mask.putInt(Math.toIntExact(feature.mask.get(j)));
//                }
//            }
//            if (padding_mask<0)
//                break;
//        }

        for (int i = 0; i < batchSize; i++) {
            AppFeature feature = features.get(batchIdx * batchSize + i);
            targetMasks.add(feature.mask);
        }

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

    private Status ConvertTrainData(String dataFile, String labelFile, String maskFile) {
        if (dataFile == null || labelFile == null || maskFile == null) {
            LOGGER.severe("dataset init failed,trainFile,idsFile,vocabFile cannot be empty");
            return Status.NULLPTR;
        }
        // read train file
        List<String> dataLines = readTxtFile(dataFile);
        List<String> labelLines = readTxtFile(labelFile);
        List<String> maskLines = readTxtFile(maskFile);
        for (int i=0; i<dataLines.size(); i++) {
            String data_str = dataLines.get(i);
            String[] data_tokens = data_str.split(" ");
            List<Integer> data = new ArrayList<>(data_tokens.length);
            for (String data_token : data_tokens){
                data.add(Float.valueOf(data_token).intValue());
            }

            String label_str = labelLines.get(i);
            if (label_str.isEmpty()) continue;

            String[] label_tokens = label_str.split(",");
            List<Integer> label = new ArrayList<>(label_tokens.length);
            for (String label_token : label_tokens){
                label.add(Integer.valueOf(label_token));
            }

            String mask_str = maskLines.get(i);
            String[] mask_tokens = mask_str.split(",");
            List<Integer> mask = new ArrayList<>(mask_tokens.length);
            for (String mask_token: mask_tokens){
                mask.add(Integer.valueOf(mask_token));
            }
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

    @Override
    public Status dataPreprocess(List<String> files) {
        String dataFile = files.get(0);
        String labelFile = files.get(1);
        String maskFile = files.get(2);
        return ConvertTrainData(dataFile, labelFile, maskFile);
    }
}
