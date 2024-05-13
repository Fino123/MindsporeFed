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

/**
 * @author Administrator
 * 类名：AutoencoderClient
 * 功能：封装执行用户画像算法的客户端行为
 */
public class AutoencoderClient extends Client {
    /** 用户画像算法的输出日志 */
    private static final Logger LOGGER = Logger.getLogger(AutoencoderClient.class.toString());
    /** 每个用户的标签数量 */
    private static final int NUM_OF_CLASS = 13;

    static {
        ClientManager.registerClient(new AutoencoderClient());
    }

    /** 聚类模型的文件路径 */
    public static String clusteringPath;
    /** 聚类模型实例 */
    private Model clusteringModel;

    /**
     * 方法名：initClusteringModel()
     * 功能：包括从模型文件中导入计算图，创建一个聚类模型实例
     * @return
     */
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

    /**
     * 方法名：initCallbacks()
     * 功能：聚类模型的回调函数，在不同模式下触发不同的回调函数
     * @param runType：模型模式，有3种：训练模式、验证模式、推理模式
     * @param dataSet：数据集
     * @return callback：回调函数列表
     */
    @Override
    public List<Callback> initCallbacks(RunType runType, DataSet dataSet) {
        boolean isSuccess = initClusteringModel();
        if(!isSuccess){
            LOGGER.info("init clustering model failed");
        }
        // 创建一个回调函数列表
        List<Callback> callbacks = new ArrayList<>();
        // 在训练模式下，往列表中加入 LossCallback() 回调函数
        if (runType == RunType.TRAINMODE) {
            Log.d("initCallbacks", "loss callback");
            Callback lossCallback = new LossCallback(model);
            callbacks.add(lossCallback);
        }
        // 在验证模式下，往列表中加入 ClusteringAccuracyCallback() 回调函数
        else if (runType == RunType.EVALMODE) {
            if (dataSet instanceof AutoencoderDataset) {
                Log.d("initCallbacks", "eval callback");
                Callback evalCallback = new ClusteringAccuracyCallback(model, clusteringModel, dataSet.batchSize, NUM_OF_CLASS, ((AutoencoderDataset) dataSet).getTargetLabels());
                callbacks.add(evalCallback);
            }
        }
        // 在推理模式下，往列表中加入 ClusteringPredictCallback() 回调函数
        else {
            Log.d("initCallbacks", "predict callback");
            Callback inferCallback = new ClusteringPredictCallback(model, clusteringModel, dataSet.batchSize, NUM_OF_CLASS);
            callbacks.add(inferCallback);
        }
        return callbacks;
    }

    /**
     * 方法名：initDataSets()
     * 功能：初始化数据集
     * @param files：数据集文件
     * @return sampleCounts：每种模式的样本数量
     */
    @Override
    public Map<RunType, Integer> initDataSets(Map<RunType, List<String>> files) {
        Map<RunType, Integer> sampleCounts = new HashMap<>(16);
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

    /**
     * 方法名：getEvalAccuracy()
     * 功能：进行模型验证，并获得准确率
     * @param evalCallbacks：进行模型验证的回调函数
     * @return acc：准确率的数值
     */
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

    /**
     * 方法名：getInferResult()
     * 功能：进行模型推理，并获得推理结果
     * @param inferCallbacks：进行模型推理的回调函数
     * @return result：推理结果，即用户标签
     */
    @Override
    public List<Object> getInferResult(List<Callback> inferCallbacks) {
        DataSet inferDataSet = dataSets.getOrDefault(RunType.INFERMODE, null);
        if (inferDataSet == null) {
            return new ArrayList<>();
        }
        for (Callback callBack : inferCallbacks) {
            if (callBack instanceof ClusteringPredictCallback) {
                List<Integer> temp = ((ClusteringPredictCallback) callBack).getPredictResults().subList(0, inferDataSet.sampleSize);
                List<Object> result = new ArrayList<>(temp.size());
                for(Integer label : temp) {
                    result.add(label.toString());
                }
                return result;
            }
        }
        LOGGER.severe("client's getEvalAccuracy() doesn't find predict related callback");
        return new ArrayList<>();
    }
}