package com.example.mindsporefederatedlearning.common;

import android.util.Log;

import com.mindspore.Model;
import com.mindspore.flclient.model.Callback;
import com.mindspore.flclient.model.Status;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * @author Administrator
 * 类名：TopkAccuracyCallback
 * 功能：APP预测算法执行验证
 */
public class TopkAccuracyCallback extends Callback {
    /** 日志 */
    private static final Logger LOGGER = Logger.getLogger(TopkAccuracyCallback.class.toString());
    /** 类别数量 */
    private final int numOfClass;
    /** 一批数据的大小 */
    private final int batchSize;
    /** 真实标签 */
    private final List<List<Integer>> targetLabels;
    /** 预测准确的数量 */
    private int correctNum;
    /** 总数 */
    private int totalNum;
    /** 预测结果 */
    private List<Boolean> results;
    /** 预测标签 */
    private List<List<Integer>> predictions;
    /** 将不使用的APP用掩码遮盖掉 */
    private List<List<Integer>> targetMasks;
    private static Map<String, List<Integer>> example = new HashMap<>();
    /** 最大的目标命中个数 */
    private int maxMatchCount;

    public static Map<String, List<Integer>> getExample() {
        if (example!=null) {
            return example;
        }
        throw new NullPointerException("example is null!");
    }

    /**
     * 构造函数
     * @param model：模型
     * @param batchSize：一批数据的大小
     * @param numOfClass：APP类别数量
     * @param targetLabels：真实标签
     * @param targetMasks：掩码
     */
    public TopkAccuracyCallback(Model model, int batchSize, int numOfClass, List<List<Integer>> targetLabels, List<List<Integer>> targetMasks) {
        super(model);
        this.batchSize = batchSize;
        this.numOfClass = numOfClass;
        this.targetLabels = targetLabels;
        this.targetMasks = targetMasks;
        results = new ArrayList<>(batchSize);
        predictions = new ArrayList<>(batchSize);
    }

    /** 获取准确率 */
    public float getAccuracy() {
        return (1.0f*correctNum)/totalNum;
    }

    /** 开始处理一批数据 */
    @Override
    public Status stepBegin() {
        Log.d("TopKAccuracy CallBack STEP BEGIN","step begin");
        return Status.SUCCESS;
    }

    /** 一批数据处理结束 */
    @Override
    public Status stepEnd() {
        model.setTrainMode(true);
        model.setLearningRate(0.0f);
        Log.d("TopKAccuracy CallBack STEP END","step end");
        maxMatchCount = Integer.MIN_VALUE;
        Status status = calAccuracy();
        if (status != Status.SUCCESS) {
            return status;
        }

        status = calClassifierResult();
        if (status != Status.SUCCESS) {
            return status;
        }

        steps++;
        model.setTrainMode(false);
        return Status.SUCCESS;
    }

    /** 对所有用户一轮处理 */
    @Override
    public Status epochBegin() {
        correctNum = 0;
        totalNum = 0;
        Log.d("EPOCH BEGIN","epoch begin");
        return Status.SUCCESS;
    }

    /** 一轮处理结束，计算准确率 */
    @Override
    public Status epochEnd() {
        Log.d("EPOCH END","epoch end");
        LOGGER.info("average accuracy:" + steps + ",acc is:" + getAccuracy());

        LOGGER.severe("match results:" + results.toString());
        LOGGER.severe("prediction results:" + predictions.toString());
        predictions.clear();
        results.clear();

        steps = 0;

        return Status.SUCCESS;
    }

    /**
     * 方法名：calClassifierResult()
     * 功能：预测一批数据的分类结果， 并计算分类准确率
     * @return status：是否执行成功
     */
    private Status calClassifierResult() {
        long startTime = System.currentTimeMillis();
        // 获取 MLP 模型的输出
        Map<String, float[]> outputs = getOutputsBySize(batchSize * numOfClass);
        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;
        Log.d("EXECUTION TIME", Long.toString(executionTime));
        if (outputs.isEmpty()) {
            LOGGER.severe("Cannot find outputs tensor for calClassifierResult");
            return Status.FAILED;
        }
        // scores 存储每个 APP 的预测概率
        Map.Entry<String, float[]> first = outputs.entrySet().iterator().next();
        float[] scores = first.getValue();

        if (scores.length != batchSize * numOfClass) {
            LOGGER.severe("Expect ClassifierResult length is:" + batchSize * numOfClass + ", but got " + scores.length);
            return Status.FAILED;
        }

        // 获取预测概率 top 4 的 APP
        for (int b=0;b<batchSize;b++){
            float[] tempScores = Arrays.copyOfRange(scores, b*numOfClass, b*numOfClass+numOfClass);
            List<Integer> mask = targetMasks.get(b + steps*batchSize);
            for (int i = 0; i < tempScores.length; i++) {
                if (!mask.contains(i)){
                    tempScores[i]=Float.MIN_VALUE;
                }
            }
            MaxHeap maxHeap = new MaxHeap(tempScores);
            int[] topk = maxHeap.getTopkIndexes(4);
            ArrayList<Integer> result = new ArrayList<>(4);
            for (int j : topk) {
                result.add(j);
            }
            predictions.add(result);
        }

        LOGGER.info("ClassifierResult is:" + predictions);
        return Status.SUCCESS;
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
        // 获取 MLP 模型的输出结果
        Map<String, float[]> outputs = getOutputsBySize(batchSize * numOfClass);
        if (outputs.isEmpty()) {
            LOGGER.severe("Cannot find outputs tensor for calAccuracy");
            return Status.FAILED;
        }
        // 获取每个 APP 的预测概率
        Map.Entry<String, float[]> first = outputs.entrySet().iterator().next();
        float[] scores = first.getValue();
        int hitCounts = 0;
        // 解析预测结果
        for (int b = 0; b < batchSize; b++) {
            float[] tempScores = Arrays.copyOfRange(scores, b*numOfClass, b*numOfClass+numOfClass);
            List<Integer> mask = targetMasks.get(b + steps*batchSize);
            // 被掩码标记的APP，其预测概率置为一个很小的数
            for (int i = 0; i < tempScores.length; i++) {
                if (!mask.contains(i)){
                    tempScores[i]=Float.MIN_VALUE;
                }
            }
            // 获取预测概率 top 4 的 APP
            MaxHeap maxHeap = new MaxHeap(tempScores);
            int[] topk = maxHeap.getTopkIndexes(4);
            boolean match = false;
            int matchNum = 0;
            // 计算这4个预测APP的命中率，即有多少个在真实标签里面
            for (int i:topk){
                if (targetLabels.get(b+steps*batchSize).contains(i)){
                    matchNum++;
                }
            }
            // 判断是否达到目标命中个数
            if (matchNum>=CommonParameter.HIT_COUNT){
                match = true;
            }
            // 计算命中率
            if (matchNum>maxMatchCount) {
                example.put("label", targetLabels.get(b + steps * batchSize));
                List<Integer> pred = new ArrayList<>();
                for (int i = 0; i < topk.length; i++) {
                    pred.add(topk[i]);
                }
                example.put("prediction", pred);
                maxMatchCount = matchNum;
            }
            if (match) {
                hitCounts++;
            }
            results.add(match);
        }
        correctNum += hitCounts;
        totalNum += batchSize;
        LOGGER.info("steps:" + steps + ",acc is:" + getAccuracy());
        return Status.SUCCESS;
    }
}
