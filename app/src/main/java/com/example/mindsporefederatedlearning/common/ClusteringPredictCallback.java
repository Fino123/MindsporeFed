package com.example.mindsporefederatedlearning.common;

import com.mindspore.MSTensor;
import com.mindspore.Model;
import com.mindspore.flclient.model.Callback;
import com.mindspore.flclient.model.Status;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * @author Administrator
 * 类名：ClusteringPredictCallback
 * 功能：用户画像模型执行推理
 */
public class ClusteringPredictCallback extends Callback {
    /** 日志 */
    private static final Logger LOGGER = Logger.getLogger(ClusteringPredictCallback.class.toString());
    /** 推理结果 */
    private final List<Integer> predictResults = new ArrayList<>();
    /** 用户类别数量 */
    private final int numOfClass;
    /** 一批数据的大小 */
    private final int batchSize;
    /** 聚类模型 */
    private Model clusteringModel;

    /**
     * 方法名：ClusteringPredictCallback()
     * 功能：执行推理，并获得聚类准确率
     * @param model：模型
     * @param clusteringModel：聚类模型
     * @param batchSize：一批数据的大小
     * @param numOfClass：用户类别数量
     */
    public ClusteringPredictCallback(Model model, Model clusteringModel, int batchSize, int numOfClass) {
        super(model);
        this.clusteringModel = clusteringModel;
        this.batchSize = batchSize;
        this.numOfClass = numOfClass;
    }

    /**
     * 方法名：getPredictResults()
     * 功能：获取预测标签
     * @return predictResults： 预测标签
     */
    public List<Integer> getPredictResults() {
        return predictResults;
    }

    /** 开始处理一批数据 */
    @Override
    public Status stepBegin() {
        return Status.SUCCESS;
    }

    /** 一批数据处理结束 */
    @Override
    public Status stepEnd() {
        Status res = calClusteringResult();
        if (res == Status.FAILED) {
            LOGGER.severe("ClusteringPredictCallback stepEnd failed");
            return Status.FAILED;
        }
        return Status.SUCCESS;

    }

    /** 对所有用户进行一轮处理 */
    @Override
    public Status epochBegin() {
        return Status.SUCCESS;
    }

    /** 一轮处理结束，计算准确率 */
    @Override
    public Status epochEnd() {
        LOGGER.info("predictCallback"+predictResults);
        return Status.SUCCESS;
    }

    /**
     * 方法名：calClusteringResult()
     * 功能：预测用户属于哪个类别（簇）
     * @return status：是否执行成功
     */
    private Status calClusteringResult() {
        // 获取AutoEncoder模型的输出结果
        List<MSTensor> outputs = this.model.getOutputsByNodeName("Default/net_with_loss-LossNetwork/_backbone-AutoEncoder/decoder-SequentialCell/0-Dense/BiasAdd-op152");
        if (outputs.isEmpty()) {
            LOGGER.severe("Cannot find outputs tensor for calClusteringResult");
            return Status.FAILED;
        }
        MSTensor out = outputs.get(0);
        float[] emb = out.getFloatData();
        // 设置聚类模型的输入数据
        List<MSTensor> clusteringInputs = clusteringModel.getInputs();
        MSTensor data = clusteringInputs.get(0);
        data.setData(emb);
        // 聚类，输出该用户与每个簇的距离
        clusteringModel.runStep();
        List<MSTensor> clusteringOutputs = clusteringModel.getOutputs();
        float[] clusteringDistances = clusteringOutputs.get(0).getFloatData();
        // 将距离最近的一个簇视为该用户的预测类别
        Integer label= 0;
        float minDist = clusteringDistances[0];
        for(int i=1; i<clusteringDistances.length; i++){
            if(clusteringDistances[i] < minDist){
                label = i;
                minDist = clusteringDistances[i];
            }
        }
        predictResults.add(label);
        LOGGER.info("steps:" + steps + ", Clustering label is:" + label);
        return Status.SUCCESS;
    }
}
