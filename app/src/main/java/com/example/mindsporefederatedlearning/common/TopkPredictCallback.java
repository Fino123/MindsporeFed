package com.example.mindsporefederatedlearning.common;

import com.mindspore.Model;
import com.mindspore.flclient.model.Callback;
import com.mindspore.flclient.model.Status;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * @author Administrator
 * 类名：TopkPredictCallback
 * 功能：APP预测算法执行推理
 */
public class TopkPredictCallback extends Callback {
    /** 日志 */
    private static final Logger LOGGER = Logger.getLogger(TopkPredictCallback.class.toString());
    /** 推理结果 */
    private final List<List<Integer>> predictResults = new ArrayList<>();
    /** APP类别数量 */
    private int numOfClass = CommonParameter.CLASS_NUM;
    /** 一批数据的大小 */
    private int batchSize = CommonParameter.batchSize;
    /** 预测概率最高的 top k 值 */
    private final int k = 4;
    /** 将不使用的APP用掩码遮盖掉 */
    private List<List<Integer>> targetMasks;

    /**
     * 构造函数
     * @param model：模型
     * @param batchSize：一批数据的大小
     * @param numOfClass：APP类别数量
     * @param targetMasks：APP掩码
     */
    public TopkPredictCallback(Model model, int batchSize, int numOfClass, List<List<Integer>> targetMasks) {
        super(model);
        this.batchSize = batchSize;
        this.numOfClass = numOfClass;
        this.targetMasks = targetMasks;
    }

    /**
     * 方法名：getTopkScoreIndex()
     * 功能：获取预测概率为 top k 的 APP 的 id 值
     * @param scores：各 APP 的预测概率
     * @param start：起始 id
     * @param end：末位 id
     * @return result：top k 的 APP 的 id 值
     */
    public List<Integer> getTopkScoreIndex(float[] scores, int start, int end) {
        List<Integer> result = new ArrayList<>();
        if (scores != null && scores.length != 0) {
            if (start < scores.length && start >= 0 && end <= scores.length && end >= 0) {
                if (scores.length<k){
                    LOGGER.severe("scores's num < k");
                    return null;
                }
                float[] tempScores = Arrays.copyOfRange(scores, start, end+1);
                MaxHeap maxHeap = new MaxHeap(tempScores);
                int[] topk = maxHeap.getTopkIndexes(k);
                for (int j : topk) {
                    result.add(j);
                }
                return result;
            } else {
                LOGGER.severe("start,end cannot out of scores length");
                return null;
            }
        } else {
            LOGGER.severe("scores cannot be empty");
            return null;
        }
    }

    /**
     * 方法名：getPredictResults()
     * 功能：获取预测标签
     * @return status： 是否执行成功
     */
    public List<List<Integer>> getPredictResults() {
        return predictResults;
    }

    @Override
    public Status stepBegin() {
        return Status.SUCCESS;
    }

    @Override
    public Status stepEnd() {
        Map<String, float[]> outputs = getOutputsBySize(batchSize * numOfClass);
        if (outputs.isEmpty()) {
            LOGGER.severe("cannot find loss tensor");
            return Status.FAILED;
        }
        Map.Entry<String, float[]> first = outputs.entrySet().iterator().next();
        float[] scores = first.getValue();
        for (int b = 0; b < batchSize; b++) {
            List<Integer> predictIdx = getTopkScoreIndex(scores, numOfClass * b, numOfClass * b + numOfClass);
            predictResults.add(predictIdx);
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
        return Status.SUCCESS;
    }
}
