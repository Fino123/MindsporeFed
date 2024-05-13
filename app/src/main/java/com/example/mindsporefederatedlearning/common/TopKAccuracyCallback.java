package com.example.mindsporefederatedlearning.common;

import android.util.Log;
import android.widget.TextView;

import com.mindspore.MSTensor;
import com.mindspore.Model;
import com.mindspore.flclient.model.Callback;
import com.mindspore.flclient.model.CommonUtils;
import com.mindspore.flclient.model.Status;

import java.lang.annotation.Target;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

public class TopKAccuracyCallback extends Callback {
    private static final Logger LOGGER = Logger.getLogger(TopKAccuracyCallback.class.toString());
    private final int numOfClass;
    private final int batchSize;
    private final List<List<Integer>> targetLabels;
    private int correct_num;
    private int total_num;
    private List<Boolean> results;
    private List<List<Integer>> predictions;
    private List<List<Integer>> targetMasks;
    private static Map<String, List<Integer>> example = new HashMap<>();
    private int maxMatchCount;

    public static Map<String, List<Integer>> getExample() {
        if (example!=null)
            return example;
        throw new NullPointerException("example is null!");
    }

    /**
     * Defining a constructor of  ClassifierAccuracyCallback.
     */
    public TopKAccuracyCallback(Model model, int batchSize, int numOfClass, List<List<Integer>> targetLabels, List<List<Integer>> targetMasks) {
        super(model);
        this.batchSize = batchSize;
        this.numOfClass = numOfClass;
        this.targetLabels = targetLabels;
        this.targetMasks = targetMasks;
        results = new ArrayList<>(batchSize);
        predictions = new ArrayList<>(batchSize);
    }

    /**
     * Get eval accuracy.
     *
     * @return accuracy.
     */
    public float getAccuracy() {
        return (1.0f*correct_num)/total_num;
    }

    @Override
    public Status stepBegin() {
        Log.d("TopKAccuracy CallBack STEP BEGIN","step begin");
        return Status.SUCCESS;
    }

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

    @Override
    public Status epochBegin() {
        correct_num = 0;
        total_num = 0;
        Log.d("EPOCH BEGIN","epoch begin");
        return Status.SUCCESS;
    }

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

    /***
     * Cal ClassifierResult for unsupervised train evaluate.
     * Now just return the mean of classifier result for saving communication data size while model update request
     * @return
     */
    private Status calClassifierResult() {
        long startTime = System.currentTimeMillis();
        Map<String, float[]> outputs = getOutputsBySize(batchSize * numOfClass);
        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;
        Log.d("EXECUTION TIME", Long.toString(executionTime));
        if (outputs.isEmpty()) {
            LOGGER.severe("Cannot find outputs tensor for calClassifierResult");
            return Status.FAILED;
        }

        Map.Entry<String, float[]> first = outputs.entrySet().iterator().next();
        float[] scores = first.getValue();

        if (scores.length != batchSize * numOfClass) {
            LOGGER.severe("Expect ClassifierResult length is:" + batchSize * numOfClass + ", but got " + scores.length);
            return Status.FAILED;
        }

        for (int b=0;b<batchSize;b++){
            float[] temp_scores = Arrays.copyOfRange(scores, b*numOfClass, b*numOfClass+numOfClass);
            List<Integer> mask = targetMasks.get(b + steps*batchSize);
            for (int i = 0; i < temp_scores.length; i++) {
                if (!mask.contains(i)){
                    temp_scores[i]=Float.MIN_VALUE;
                }
            }
            MaxHeap maxHeap = new MaxHeap(temp_scores);
            int[] topk = maxHeap.getTopKIndexes(4);
            ArrayList<Integer> result = new ArrayList<>(4);
            for (int j : topk) result.add(j);
            predictions.add(result);
        }

        LOGGER.info("ClassifierResult is:" + predictions);
        return Status.SUCCESS;
    }


    private Status calAccuracy() {
        if (targetLabels == null || targetLabels.isEmpty()) {
            LOGGER.severe("labels cannot be null");
            return Status.NULLPTR;
        }
        Map<String, float[]> outputs = getOutputsBySize(batchSize * numOfClass);
        if (outputs.isEmpty()) {
            LOGGER.severe("Cannot find outputs tensor for calAccuracy");
            return Status.FAILED;
        }
        Map.Entry<String, float[]> first = outputs.entrySet().iterator().next();
        float[] scores = first.getValue();
        int hitCounts = 0;
        for (int b = 0; b < batchSize; b++) {
            float[] temp_scores = Arrays.copyOfRange(scores, b*numOfClass, b*numOfClass+numOfClass);
            List<Integer> mask = targetMasks.get(b + steps*batchSize);
            for (int i = 0; i < temp_scores.length; i++) {
                if (!mask.contains(i)){
                    temp_scores[i]=Float.MIN_VALUE;
                }
            }
            MaxHeap maxHeap = new MaxHeap(temp_scores);
            int[] topk = maxHeap.getTopKIndexes(4);
            boolean match = false;
            int match_num = 0;
            for (int i:topk){
                if (targetLabels.get(b+steps*batchSize).contains(i)){
                    match_num++;
                }
            }
            if (match_num>=CommonParameter.HIT_COUNT){
                match = true;
            }
            if (match_num>maxMatchCount) {
                example.put("label", targetLabels.get(b + steps * batchSize));
                List<Integer> pred = new ArrayList<>();
                for (int i = 0; i < topk.length; i++) {
                    pred.add(topk[i]);
                }
                example.put("prediction", pred);
                maxMatchCount = match_num;
            }
            if (match) hitCounts++;
            results.add(match);
        }
        correct_num += hitCounts;
        total_num += batchSize;
        LOGGER.info("steps:" + steps + ",acc is:" + getAccuracy());
        return Status.SUCCESS;
    }
}
