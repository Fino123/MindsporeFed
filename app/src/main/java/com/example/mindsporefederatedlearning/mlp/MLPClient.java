package com.example.mindsporefederatedlearning.mlp;

import android.os.Trace;
import android.telecom.Call;

import com.example.mindsporefederatedlearning.common.CommonParameter;
import com.example.mindsporefederatedlearning.common.TopKAccuracyCallback;
import com.example.mindsporefederatedlearning.common.TopKPredictCallback;
import com.mindspore.flclient.model.Callback;
import com.mindspore.flclient.model.Client;
import com.mindspore.flclient.model.ClientManager;
import com.mindspore.flclient.model.DataSet;
import com.mindspore.flclient.model.LossCallback;
import com.mindspore.flclient.model.RunType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

public class MLPClient extends Client {
    private static final Logger LOGGER = Logger.getLogger(MLPClient.class.toString());

    static {
        ClientManager.registerClient(new MLPClient());
    }

    @Override
    public List<Callback> initCallbacks(RunType runType, DataSet dataSet) {
        List<Callback> callbacks = new ArrayList<>();
        if (runType == RunType.TRAINMODE) {
            Callback lossCallback = new LossCallback(model);
            callbacks.add(lossCallback);
        } else if (runType == RunType.EVALMODE) {
            if (dataSet instanceof MLPDataset) {
                model.setTrainMode(true);
                model.setLearningRate(0.0f);
                Callback callback = new TopKAccuracyCallback(model, dataSet.batchSize, CommonParameter.CLASS_NUM, ((MLPDataset) dataSet).getTargetLabels(), ((MLPDataset) dataSet).getTargetMasks());
                callbacks.add(callback);
            }
        } else {
            Callback inferCallback = new TopKPredictCallback(model, dataSet.batchSize, CommonParameter.CLASS_NUM, ((MLPDataset) dataSet).getTargetMasks());
            callbacks.add(inferCallback);
        }
        return callbacks;
    }

    @Override
    public Map<RunType, Integer> initDataSets(Map<RunType, List<String>> files) {

        Map<RunType, Integer> sampleCounts = new HashMap<>();
        List<String> trainFiles = files.getOrDefault(RunType.TRAINMODE, null);
        if (trainFiles != null) {
            DataSet trainDataSet = new MLPDataset(RunType.TRAINMODE, CommonParameter.batchSize);
            trainDataSet.init(trainFiles);
            dataSets.put(RunType.TRAINMODE, trainDataSet);
            sampleCounts.put(RunType.TRAINMODE, trainDataSet.sampleSize);
        }
        List<String> evalFiles = files.getOrDefault(RunType.EVALMODE, null);
        if (evalFiles != null) {
            DataSet evalDataSet = new MLPDataset(RunType.EVALMODE, CommonParameter.batchSize);
            evalDataSet.init(evalFiles);
            dataSets.put(RunType.EVALMODE, evalDataSet);
            sampleCounts.put(RunType.EVALMODE, evalDataSet.sampleSize);
        }
        List<String> inferFiles = files.getOrDefault(RunType.INFERMODE, null);
        if (inferFiles != null) {
            DataSet evalDataSet = new MLPDataset(RunType.INFERMODE, CommonParameter.batchSize);
            evalDataSet.init(evalFiles);
            dataSets.put(RunType.INFERMODE, evalDataSet);
            sampleCounts.put(RunType.INFERMODE, evalDataSet.sampleSize);
        }
        return sampleCounts;
    }

    @Override
    public float getEvalAccuracy(List<Callback> evalCallbacks) {
        for (Callback callBack : evalCallbacks) {
            if (callBack instanceof TopKAccuracyCallback) {
                return ((TopKAccuracyCallback) callBack).getAccuracy();
            }
        }
        LOGGER.severe("don not find accuracy related callback");
        return Float.NaN;
    }

    @Override
    public List<Object> getInferResult(List<Callback> inferCallbacks) {
        DataSet inferDataSet = dataSets.getOrDefault(RunType.INFERMODE, null);
        if (inferDataSet == null) {
            return new ArrayList<>();
        }
        for (Callback callBack : inferCallbacks) {
            if (callBack instanceof TopKPredictCallback) {
                List<List<Integer>> temp = ((TopKPredictCallback) callBack).getPredictResults().subList(0, inferDataSet.sampleSize);
                List<Object> result = new ArrayList<>(temp.size());
                for(List<Integer> list :temp){
                    result.add(list.toString());
                }
                return result;
            }
        }
        LOGGER.severe("don not find accuracy related callback");
        return new ArrayList<>();
    }

}
