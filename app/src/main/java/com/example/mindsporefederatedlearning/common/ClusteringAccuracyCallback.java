package com.example.mindsporefederatedlearning.common;

import android.util.Log;

import com.mindspore.MSTensor;
import com.mindspore.Model;
import com.mindspore.flclient.model.Callback;
import com.mindspore.flclient.model.Status;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

/**
 * @author Administrator
 * 类名：ClusteringAccuracyCallback
 * 功能：用户画像模型执行验证
 */
public class ClusteringAccuracyCallback extends Callback {
    /** 日志 */
    private static final Logger LOGGER = Logger.getLogger(ClusteringAccuracyCallback.class.toString());
    /** 用户类别数量 */
    private final int numOfClass;
    /** 一批数据的大小 */
    private final int batchSize;
    /** 真实标签 */
    private final List<List<Integer>> targetLabels;
    /** 预测准确率 */
    private List<Float> accResults;
    /** 预测结果 */
    private List<Integer> predictions;
    /** 准确率 */
    private float accuracy;
    /** 聚类模型 */
    private Model clusteringModel;
    /** 单个用户的真实标签 */
    private static List<Integer> oneUserLabels;
    /** 单个用户的预测标签 */
    private static List<Integer> onePredictedLabels;

    /**
     * 方法名：getOneUserLabels()
     * 功能：获取单个用户的真实标签
     * @return oneUserLabels：真实标签
     */
    public static List<Integer> getOneUserLabels() {
        return oneUserLabels;
    }

    /**
     * 方法名：getOnePredictedLabels()
     * 功能：获取单个用户的预测标签
     * @return onePredictedLabels：预测标签
     */
    public static List<Integer> getOnePredictedLabels() {
        return onePredictedLabels;
    }


    /**
     * 构造函数
     * 功能：执行推理，并获得聚类准确率
     * @param model：模型
     * @param clusteringModel：聚类模型
     * @param batchSize：一批数据的大小
     * @param numOfClass：用户类别数量
     * @param targetLabels：真实标签
     */
    public ClusteringAccuracyCallback(Model model, Model clusteringModel, int batchSize, int numOfClass, List<List<Integer>> targetLabels) {
        super(model);
        this.clusteringModel = clusteringModel;
        this.batchSize = batchSize;
        this.numOfClass = numOfClass;
        this.targetLabels = targetLabels;
        // record acc for each user for each batch
        accResults = new ArrayList<>(batchSize);
        // record predicted label for each user
        predictions = new ArrayList<>(batchSize);
    }

    /** 获取准确率 */
    public float getAccuracy() {
        return accuracy;
    }

    /** 开始处理一批数据 */
    @Override
    public Status stepBegin() {
        Log.d("ClusteringAccuracy Callback STEP BEGIN","step begin");
        return Status.SUCCESS;
    }

    /** 一批数据处理结束 */
    @Override
    public Status stepEnd() {
        Status status = calAccuracy();
        if (status != Status.SUCCESS) {
            return status;
        }

        status = calClusteringResult();
        if (status != Status.SUCCESS) {
            return status;
        }

        steps++;
        return Status.SUCCESS;
    }

    /** 对所有用户一轮处理 */
    @Override
    public Status epochBegin() {
        Log.d("EPOCH BEGIN","epoch begin");
        return Status.SUCCESS;
    }

    /** 一轮处理结束，计算准确率 */
    @Override
    public Status epochEnd() {
        Log.d("EPOCH END","epoch end");
        LOGGER.info("average accuracy from step 0 to step " + steps + " is:" + accuracy / steps);
        accuracy = accuracy / steps;

        LOGGER.severe("Acc Callback, prediction acc:" + accResults.toString());
        LOGGER.severe("Acc Callback, prediction label:" + predictions.toString());
        predictions.clear();
        accResults.clear();

        steps = 0;
        return Status.SUCCESS;
    }

    /** 用户聚类成5个簇，每个簇中心的标签 */
    private static Integer[][] groupLabels = {
            {3, 5, 4, 2, 0, 5, 3, 6, 4, 15, 5, 3, 6},
            {6, 3, 4, 2, 15, 6, 3, 4, 15, 9, 6, 3, 4},
            {3, 5, 4, 2, 0, 3, 5, 6, 4, 15, 5, 6, 3},
            {3, 6, 4, 2, 0, 6, 3, 4, 2 ,15, 6, 3, 4},
            {3, 6, 4, 2, 0, 6, 3, 4, 15, 2, 6, 3, 4}};

    /**
     * 方法名：getClusteringLabel()
     * 功能：在 AutoEncoder 模型获得用户表征后，执行用户聚类，预测用户类别
     * @param userEmbedding：用户表征
     * @return label：用户所属的类别
     */
    public Integer getClusteringLabel(MSTensor userEmbedding){
        // 将用户表征作为聚类模型的输入
        long startTime = System.currentTimeMillis();
        List<MSTensor> clusteringInputs = clusteringModel.getInputs();
        MSTensor data = clusteringInputs.get(0);
        float[] emb = userEmbedding.getFloatData();
        data.setData(emb);
        // 聚类模型输出该用户与每个簇的距离
        clusteringModel.runStep();
        List<MSTensor> clusteringOutputs = clusteringModel.getOutputs();
        float[] clusteringDistances = clusteringOutputs.get(0).getFloatData();
        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;
        Log.d("EXECUTION TIME", Long.toString(executionTime));
        // 将距离最近的一个簇视为该用户的预测类别
        Integer label= 0;
        float minDist = clusteringDistances[0];
        for(int i=1; i<clusteringDistances.length; i++){
            if(clusteringDistances[i] < minDist){
                label = i;
                minDist = clusteringDistances[i];
            }
        }
        return label;
    }


    /**
     * 方法名：
     * 功能：在 AutoEncoder 模型获得用户表征后，执行用户聚类，预测用户标签并计算准确率
     * @param userEmbedding：用户表征
     * @return acc：预测标签的准确率
     */
    public float getClusteringAcc(MSTensor userEmbedding){
        // 将用户表征作为聚类模型的输入
        List<MSTensor> clusteringInputs = clusteringModel.getInputs();
        MSTensor data = clusteringInputs.get(0);
        float[] emb = userEmbedding.getFloatData();
        data.setData(emb);
        // 聚类模型输出该用户与每个簇的距离
        clusteringModel.runStep();
        List<MSTensor> clusteringOutputs = clusteringModel.getOutputs();
        float[] clusteringDistances = clusteringOutputs.get(0).getFloatData();
        // 将距离最近的一个簇视为该用户的预测类别
        int label= 0;
        float minDist = clusteringDistances[0];
        for(int i=1; i<clusteringDistances.length; i++){
            if(clusteringDistances[i] < minDist){
                label = i;
                minDist = clusteringDistances[i];
            }
        }
        // 获取该类别的簇类标签，作为该用户的预测标签，计算预测标签与真实标签的匹配率，作为预测准确率
        // 每个用户的标签均有13个，第1~5个为粘性app标签，计算该组标签的匹配率
        int hit = 0;
        int labelStartId = 0;
        int labelEndId = 5;
        List<Integer> targetSubLabels = targetLabels.get(0).subList(labelStartId, labelEndId);
        for(int j=labelStartId; j<labelEndId; j++){
            if(targetSubLabels.contains(groupLabels[label][j])) {
                hit += 1;
            }
        }
        // 第6~10个为高频使用app标签，计算该组标签的匹配率
        labelStartId = 5;
        labelEndId = 10;
        targetSubLabels = targetLabels.get(0).subList(labelStartId,labelEndId);
        for(int j=labelStartId; j<labelEndId; j++){
            if(targetSubLabels.contains(groupLabels[label][j])) {
                hit += 1;
            }
        }
        // 第11~13个为长期使用app标签，计算该组标签的匹配率
        labelStartId = 10;
        labelEndId = 13;
        targetSubLabels = targetLabels.get(0).subList(labelStartId,labelEndId);
        for(int j=labelStartId; j<labelEndId; j++){
            if(targetSubLabels.contains(groupLabels[label][j])) {
                hit += 1;
            }
        }
        // 计算准确率
        return (float) hit / labelEndId;
    }


    /**
     * 方法名：calAccuracy()
     * 功能：对一批数据进行预测，计算准确率
     * @return status：是否执行成功
     */
    private Status calAccuracy() {
        if (targetLabels == null || targetLabels.isEmpty()) {
            LOGGER.severe("labels cannot be null");
            return Status.NULLPTR;
        }
        // 获取AutoEncoder模型输出结果
        List<MSTensor> outputs = this.model.getOutputsByNodeName("Default/net_with_loss-LossNetwork/_backbone-AutoEncoder/decoder-SequentialCell/0-Dense/BiasAdd-op152");
        if (outputs.isEmpty()) {
            LOGGER.severe("Cannot find outputs tensor for calAccuracy");
            return Status.FAILED;
        }
        if(batchSize != 1) {
            LOGGER.info("batchsize should be 1");
        }
        MSTensor out = outputs.get(0);
        // 调用 getClusteringAcc()，执行聚类，并获得预测标签的准确率
        float acc = getClusteringAcc(out);
        // 先存储该数据的准确率，等该 epoch 结束后再求平均值
        accuracy += acc;
        accResults.add(acc);
        return Status.SUCCESS;
    }

    /**
     * 方法名：calClusteringResult()
     * 功能：计算聚类结果
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
        // 获取聚类之后得到的预测标签
        Integer predictLabel = getClusteringLabel(out);
        predictions.add(predictLabel);

        // 对第一个数据进行预测
        if (steps == 0){
            List<MSTensor> inputs = this.model.getInputs();
            Log.d("input size", inputs.get(1).getShape()[0]+" "+inputs.get(1).getShape()[1]);
            MSTensor labelTensor = inputs.get(1);
            float[] userLabels = labelTensor.getFloatData();

            MaxHeap stickyLabels = new MaxHeap(Arrays.copyOf(userLabels,35));
            MaxHeap frequentLabels = new MaxHeap(Arrays.copyOfRange(userLabels, 35, 70));
            MaxHeap longTermLabels = new MaxHeap(Arrays.copyOfRange(userLabels, 70, 105));
            int[] stickyId = stickyLabels.getTopkIndexes(5);
            int[] frequentId = frequentLabels.getTopkIndexes(5);
            int[] longtermId = longTermLabels.getTopkIndexes(3);

            List<Integer> trueLabels = new ArrayList<>();
            for(int i=0; i<stickyId.length; i++){
                trueLabels.add(stickyId[i]);
            }
            for(int i=0; i<frequentId.length; i++){
                trueLabels.add(frequentId[i]);
            }
            for(int i=0; i<longtermId.length; i++){
                trueLabels.add(longtermId[i]);
            }

            List<Integer> predictedLabels = new ArrayList<>();
            for(int i=0; i<trueLabels.size(); i++){
                predictedLabels.add(groupLabels[predictLabel][i]);
            }
            oneUserLabels = trueLabels;
            onePredictedLabels = predictedLabels;
        }
        return Status.SUCCESS;
    }
}
