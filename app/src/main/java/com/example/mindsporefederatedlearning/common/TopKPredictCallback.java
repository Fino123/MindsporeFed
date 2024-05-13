package com.example.mindsporefederatedlearning.common;

import com.mindspore.Model;
import com.mindspore.flclient.model.Callback;
import com.mindspore.flclient.model.Status;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Stack;
import java.util.logging.Logger;

public class TopKPredictCallback extends Callback {
    private static final Logger LOGGER = Logger.getLogger(TopKPredictCallback.class.toString());
    private final List<List<Integer>> predictResults = new ArrayList<>();
    private int numOfClass = CommonParameter.CLASS_NUM;
    private int batchSize = CommonParameter.batchSize;
    private final int k = 4;

    private List<List<Integer>> targetMasks;

    /**
     * Defining a constructor of predict callback.
     */
    public TopKPredictCallback(Model model, int batchSize, int numOfClass, List<List<Integer>> targetMasks) {
        super(model);
        this.batchSize = batchSize;
        this.numOfClass = numOfClass;
        this.targetMasks = targetMasks;
    }

    public List<Integer> getTopKScoreIndex(float[] scores, int start, int end) {
        List<Integer> result = new ArrayList<>();
        if (scores != null && scores.length != 0) {
            if (start < scores.length && start >= 0 && end <= scores.length && end >= 0) {
                if (scores.length<k){
                    LOGGER.severe("scores's num < k");
                    return null;
                }
                float[] temp_scores = Arrays.copyOfRange(scores, start, end+1);
                MaxHeap maxHeap = new MaxHeap(temp_scores);
                int[] topk = maxHeap.getTopKIndexes(k);
                for (int j : topk) result.add(j);
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
     * Get predict results.
     *
     * @return predict result.
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
            List<Integer> predictIdx = getTopKScoreIndex(scores, numOfClass * b, numOfClass * b + numOfClass);
            predictResults.add(predictIdx);
        }
        return Status.SUCCESS;
    }

    @Override
    public Status epochBegin() {
        return Status.SUCCESS;
    }

    @Override
    public Status epochEnd() {
        return Status.SUCCESS;
    }
}
