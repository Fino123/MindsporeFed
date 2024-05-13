package com.example.mindsporefederatedlearning.mlp;

import com.example.mindsporefederatedlearning.common.CommonParameter;
import com.example.mindsporefederatedlearning.common.TopkAccuracyCallback;
import com.example.mindsporefederatedlearning.common.TopkPredictCallback;
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
 * 类名：MlpClient
 * 功能：封装执行APP预测算法的客户端行为
 */
public class MlpClient extends Client {
    /** APP预测算法的输出日志 */
    private static final Logger LOGGER = Logger.getLogger(MlpClient.class.toString());

    static {
        ClientManager.registerClient(new MlpClient());
    }

    /**
     * 方法名：initCallbacks()
     * 功能：MLP模型的回调函数，在不同模式下触发不同的回调函数
     * @param runType：模型模式，有3种：训练模式、验证模式、推理模式
     * @param dataSet：数据集
     * @return callback：回调函数列表
     */
    @Override
    public List<Callback> initCallbacks(RunType runType, DataSet dataSet) {
        // 创建一个回调函数列表
        List<Callback> callbacks = new ArrayList<>();
        // 在训练模式下，往列表中加入 LossCallback() 回调函数
        if (runType == RunType.TRAINMODE) {
            Callback lossCallback = new LossCallback(model);
            callbacks.add(lossCallback);
        }
        // 在验证模式下，往列表中加入 TopkAccuracyCallback() 回调函数
        else if (runType == RunType.EVALMODE) {
            if (dataSet instanceof MlpDataset) {
                model.setTrainMode(true);
                model.setLearningRate(0.0f);
                Callback callback = new TopkAccuracyCallback(model, dataSet.batchSize, CommonParameter.CLASS_NUM, ((MlpDataset) dataSet).getTargetLabels(), ((MlpDataset) dataSet).getTargetMasks());
                callbacks.add(callback);
            }
        }
        // 在推理模式下，往列表中加入 TopkPredictCallback() 回调函数
        else {
            Callback inferCallback = new TopkPredictCallback(model, dataSet.batchSize, CommonParameter.CLASS_NUM, ((MlpDataset) dataSet).getTargetMasks());
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
            DataSet trainDataSet = new MlpDataset(RunType.TRAINMODE, CommonParameter.batchSize);
            trainDataSet.init(trainFiles);
            dataSets.put(RunType.TRAINMODE, trainDataSet);
            sampleCounts.put(RunType.TRAINMODE, trainDataSet.sampleSize);
        }
        List<String> evalFiles = files.getOrDefault(RunType.EVALMODE, null);
        if (evalFiles != null) {
            DataSet evalDataSet = new MlpDataset(RunType.EVALMODE, CommonParameter.batchSize);
            evalDataSet.init(evalFiles);
            dataSets.put(RunType.EVALMODE, evalDataSet);
            sampleCounts.put(RunType.EVALMODE, evalDataSet.sampleSize);
        }
        List<String> inferFiles = files.getOrDefault(RunType.INFERMODE, null);
        if (inferFiles != null) {
            DataSet evalDataSet = new MlpDataset(RunType.INFERMODE, CommonParameter.batchSize);
            evalDataSet.init(evalFiles);
            dataSets.put(RunType.INFERMODE, evalDataSet);
            sampleCounts.put(RunType.INFERMODE, evalDataSet.sampleSize);
        }
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
            if (callBack instanceof TopkAccuracyCallback) {
                return ((TopkAccuracyCallback) callBack).getAccuracy();
            }
        }
        LOGGER.severe("don not find accuracy related callback");
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
            if (callBack instanceof TopkPredictCallback) {
                List<List<Integer>> temp = ((TopkPredictCallback) callBack).getPredictResults().subList(0, inferDataSet.sampleSize);
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
