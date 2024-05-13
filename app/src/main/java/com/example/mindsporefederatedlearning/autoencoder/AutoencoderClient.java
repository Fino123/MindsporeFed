package com.example.mindsporefederatedlearning.autoencoder;

import android.util.Log;

import com.example.mindsporefederatedlearning.common.ClusteringAccuracyCallback;
import com.example.mindsporefederatedlearning.common.ClusteringPredictCallback;
import com.mindspore.Graph;
import com.mindspore.Model;
import com.mindspore.config.DeviceType;
import com.mindspore.config.MSContext;
import com.mindspore.config.TrainCfg;
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

public class AutoencoderClient extends Client {
    private static final Logger LOGGER = Logger.getLogger(AutoencoderClient.class.toString());
    private static final int NUM_OF_CLASS = 13;

    static {
        ClientManager.registerClient(new AutoencoderClient());
    }

    public static String clusteringPath;
    private Model clusteringModel;
    public boolean initClusteringModel() {
        MSContext context = new MSContext();
        context.init();
        context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
        TrainCfg trainCfg = new TrainCfg();
        trainCfg.init();
        Graph graph = new Graph();
        graph.load(clusteringPath);
        clusteringModel = new Model();
        boolean isSuccess = clusteringModel.build(graph, context, trainCfg);
        Log.d("clustering", "model build success："+isSuccess);
        isSuccess = clusteringModel.setupVirtualBatch(1, 0.00f, 0.00f);
        return isSuccess;
    }

    @Override
    public List<Callback> initCallbacks(RunType runType, DataSet dataSet) {
        boolean isSuccess = initClusteringModel();
        if(!isSuccess){
            LOGGER.info("init clustering model failed");
        }

        List<Callback> callbacks = new ArrayList<>();
        if (runType == RunType.TRAINMODE) {
            Log.d("initCallbacks", "loss callback");
            Callback lossCallback = new LossCallback(model);
            callbacks.add(lossCallback);
        } else if (runType == RunType.EVALMODE) {
            if (dataSet instanceof AutoencoderDataset) {
                Log.d("initCallbacks", "eval callback");
                Callback evalCallback = new ClusteringAccuracyCallback(model, clusteringModel, dataSet.batchSize, NUM_OF_CLASS, ((AutoencoderDataset) dataSet).getTargetLabels());
                callbacks.add(evalCallback);
            }
        } else {
            Log.d("initCallbacks", "predict callback");
            Callback inferCallback = new ClusteringPredictCallback(model, clusteringModel, dataSet.batchSize, NUM_OF_CLASS);
            callbacks.add(inferCallback);
        }
        return callbacks;
    }

    @Override
    public Map<RunType, Integer> initDataSets(Map<RunType, List<String>> files) {
        Map<RunType, Integer> sampleCounts = new HashMap<>();
        List<String> trainFiles = files.getOrDefault(RunType.TRAINMODE, null);
        if (trainFiles != null) {
            Log.d("init datasets", "init train files");
            DataSet trainDataSet = new AutoencoderDataset(RunType.TRAINMODE, 1);
            trainDataSet.init(trainFiles);
            dataSets.put(RunType.TRAINMODE, trainDataSet);
            sampleCounts.put(RunType.TRAINMODE, trainDataSet.sampleSize);
        }
        List<String> evalFiles = files.getOrDefault(RunType.EVALMODE, null);
        if (evalFiles != null) {
            Log.d("init datasets", "init eval files");
            evalFiles = files.getOrDefault(RunType.EVALMODE, null);
            DataSet evalDataSet = new AutoencoderDataset(RunType.EVALMODE, 1);
            evalDataSet.init(evalFiles);
            dataSets.put(RunType.EVALMODE, evalDataSet);
            sampleCounts.put(RunType.EVALMODE, evalDataSet.sampleSize);
        }
        List<String> inferFiles = files.getOrDefault(RunType.INFERMODE, null);
        if (inferFiles != null) {
            Log.d("init datasets", "init test files");
            DataSet evalDataSet = new AutoencoderDataset(RunType.INFERMODE, 1);
            evalDataSet.init(evalFiles);
            dataSets.put(RunType.INFERMODE, evalDataSet);
            sampleCounts.put(RunType.INFERMODE, evalDataSet.sampleSize);
        }
        Log.d("init datasets", "init finished");
        return sampleCounts;
    }

    @Override
    public float getEvalAccuracy(List<Callback> evalCallbacks) {
        for (Callback callBack : evalCallbacks) {
            if (callBack instanceof ClusteringAccuracyCallback) {
                return ((ClusteringAccuracyCallback) callBack).getAccuracy();
            }
        }
        LOGGER.severe("client's getEvalAccuracy() doesn't find accuracy related callback");
        return Float.NaN;
    }

    @Override
    public List<Object> getInferResult(List<Callback> inferCallbacks) {
        ///*
        DataSet inferDataSet = dataSets.getOrDefault(RunType.INFERMODE, null);
        if (inferDataSet == null) {
            return new ArrayList<>();
        }
        for (Callback callBack : inferCallbacks) {
            if (callBack instanceof ClusteringPredictCallback) {
                List<Integer> temp = ((ClusteringPredictCallback) callBack).getPredictResults().subList(0, inferDataSet.sampleSize);
                List<Object> result = new ArrayList<>(temp.size());
                for(Integer label : temp){
                    result.add(label.toString());
                }
                return result;
            }
        }
        LOGGER.severe("client's getEvalAccuracy() doesn't find predict related callback");
        return new ArrayList<>();
    }
}