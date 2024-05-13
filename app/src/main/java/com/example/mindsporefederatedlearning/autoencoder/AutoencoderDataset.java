
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


public class AutoencoderDataset extends DataSet {
    private static final Logger LOGGER = Logger.getLogger(AutoencoderDataset.class.toString());
    private final RunType runType;
    private List<UserFeature> features;
    private List<List<Integer>> targetLabels;

    public AutoencoderDataset(RunType runType, int batch_size){
        this.runType = runType;
        this.batchSize = batch_size;
        this.features = new ArrayList<>();
    }

    public List<List<Integer>> getTargetLabels() {
        return targetLabels;
    }


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


    private Status ConvertData(String dataFile, String labelFile){
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
            String data_str = dataLines.get(i);
            String[] data_tokens = data_str.split(" ");
            List<Float> data = new ArrayList<>(data_tokens.length);
            for (String data_token : data_tokens){
                data.add(Float.valueOf(data_token));
            }
            // parsing labels
            String label_str = labelLines.get(i);
            String[] label_tokens = label_str.split(" ");
            List<Integer> labels = new ArrayList<>(label_tokens.length);
            for (String label_token : label_tokens){
                labels.add(Integer.valueOf(label_token));
            }
            features.add(new UserFeature(data, labels));
        }
        sampleSize = features.size();
        Log.d("convert train data", "finished");
        return Status.SUCCESS;
    }


    @Override
    public void fillInputBuffer(List<ByteBuffer> inputsBuffer, int batchIdx) {
        for (ByteBuffer inputBuffer : inputsBuffer) {
            inputBuffer.clear();
        }

        ByteBuffer input = inputsBuffer.get(0);
        input.order(ByteOrder.nativeOrder());

        ByteBuffer label = inputsBuffer.get(1);
        label.order(ByteOrder.nativeOrder());

        for (int i = 0; i < batchSize; i++) {
            UserFeature feature = features.get(batchIdx * batchSize + i);
            for (int j = 0; j < feature.data.size(); j++) {
                input.putFloat(feature.data.get(j));
                label.putFloat(feature.data.get(j));        // Note that the input of Autoencoder is (data,data), not (data,label)
            }
            targetLabels.add(feature.labels);                // this label is used for kmeans
        }
    }

    @Override
    public void shuffle() {

    }

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


    @Override
    public Status dataPreprocess(List<String> files) {
        //return Status.SUCCESS;
        String dataFile = files.get(0);
        String labelFile = files.get(1);
        return ConvertData(dataFile, labelFile);
    }

}

