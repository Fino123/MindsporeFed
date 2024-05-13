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

public class ClusteringAccuracyCallback extends Callback {
    private static final Logger LOGGER = Logger.getLogger(ClusteringAccuracyCallback.class.toString());
    private final int numOfClass;
    private final int batchSize;
    private final List<List<Integer>> targetLabels;
    private List<Float> accResults;
    private List<Integer> predictions;
    private float accuracy;

    private Model clusteringModel;

    public static List<Integer> getOne_user_labels() {
        return one_user_labels;
    }

    private static List<Integer> one_user_labels;

    public static List<Integer> getOne_predicted_labels() {
        return one_predicted_labels;
    }

    private static List<Integer> one_predicted_labels;


    public ClusteringAccuracyCallback(Model model, Model clustering_model, int batchSize, int numOfClass, List<List<Integer>> targetLabels) {
        super(model);
        this.clusteringModel = clustering_model;
        this.batchSize = batchSize;
        this.numOfClass = numOfClass;
        this.targetLabels = targetLabels;
        accResults = new ArrayList<>(batchSize);    // record acc for each user for each batch
        predictions = new ArrayList<>(batchSize);   // record predicted label for each user
    }


    public float getAccuracy() {
        return accuracy;
    }


    @Override
    public Status stepBegin() {
        Log.d("ClusteringAccuracy Callback STEP BEGIN","step begin");
        return Status.SUCCESS;
    }

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

    @Override
    public Status epochBegin() {
        Log.d("EPOCH BEGIN","epoch begin");
        return Status.SUCCESS;
    }

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

    /****************  link autoencoder with kmeans  ******************/
    private static Integer[][] groupLabels = {
            {3, 5, 4, 2, 0, 5, 3, 6, 4, 15, 5, 3, 6},
            {6, 3, 4, 2, 15, 6, 3, 4, 15, 9, 6, 3, 4},
            {3, 5, 4, 2, 0, 3, 5, 6, 4, 15, 5, 6, 3},
            {3, 6, 4, 2, 0, 6, 3, 4, 2 ,15, 6, 3, 4},
            {3, 6, 4, 2, 0, 6, 3, 4, 15, 2, 6, 3, 4}};

    private static final String[] app_categories = {"game", "finance", "video_players", "communication", "social", "others", "transsion", "maps_and_navigation", "books_and_reference", "tools", "lifestyle", "dating", "productivity", "personalization", "business", "photography", "music_and_audio", "shopping", "entertainment", "education", "health_and_fitness", "travel_and_local", "sports", "news_and_magazines", "food_and_drink", "art_and_design", "weather", "medical", "parenting", "events", "beauty", "house_and_home", "auto_and_vehicles", "libraries_and_demo", "comics"};


    public Integer getClusteringLabel(MSTensor userEmbedding){
        // load kmeans model and feed into embedding, then get predicted label [distance]
        long startTime = System.currentTimeMillis();
        List<MSTensor> clustering_inputs = clusteringModel.getInputs();
        MSTensor data = clustering_inputs.get(0);
        float[] emb = userEmbedding.getFloatData();
        data.setData(emb);
        // run step
        clusteringModel.runStep();
        List<MSTensor> clustering_outputs = clusteringModel.getOutputs();
        float[] clustering_distances = clustering_outputs.get(0).getFloatData();
        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;
        Log.d("EXECUTION TIME", Long.toString(executionTime));
        // parsing clustering result
        Integer label= 0;
        float min_dist = clustering_distances[0];
        for(int i=1; i<clustering_distances.length; i++){
            if(clustering_distances[i] < min_dist){
                label = i;
                min_dist = clustering_distances[i];
            }
        }
        return label;
    }


    public float getClusteringAcc(MSTensor userEmbedding){
        // set input data
        List<MSTensor> clustering_inputs = clusteringModel.getInputs();
        MSTensor data = clustering_inputs.get(0);
        float[] emb = userEmbedding.getFloatData();
        data.setData(emb);
        // run step
        clusteringModel.runStep();
        List<MSTensor> clustering_outputs = clusteringModel.getOutputs();
        float[] clustering_distances = clustering_outputs.get(0).getFloatData();
        // parsing clustering result
        Integer label= 0;
        float min_dist = clustering_distances[0];
        for(int i=1; i<clustering_distances.length; i++){
            if(clustering_distances[i] < min_dist){
                label = i;
                min_dist = clustering_distances[i];
            }
        }
        int hit = 0;
        List<Integer> targetSubLabels = targetLabels.get(0).subList(0,5);
        for(int j=0; j<5; j++){
            if(targetSubLabels.contains(groupLabels[label][j]))
                hit += 1;
        }
        targetSubLabels = targetLabels.get(0).subList(5,10);
        for(int j=5; j<10; j++){
            if(targetSubLabels.contains(groupLabels[label][j]))
                hit += 1;
        }
        targetSubLabels = targetLabels.get(0).subList(10,13);
        for(int j=10; j<13; j++){
            if(targetSubLabels.contains(groupLabels[label][j]))
                hit += 1;
        }
        return (float) hit / 13;
    }


    private Status calAccuracy() {
        if (targetLabels == null || targetLabels.isEmpty()) {
            LOGGER.severe("labels cannot be null");
            return Status.NULLPTR;
        }
        List<MSTensor> outputs = this.model.getOutputsByNodeName("Default/net_with_loss-LossNetwork/_backbone-AutoEncoder/decoder-SequentialCell/0-Dense/BiasAdd-op152");
        if (outputs.isEmpty()) {
            LOGGER.severe("Cannot find outputs tensor for calAccuracy");
            return Status.FAILED;
        }
        if(batchSize != 1){
            LOGGER.info("batchsize should be 1");
        }
        MSTensor out = outputs.get(0);
        float acc = getClusteringAcc(out);
        accuracy += acc;
        accResults.add(acc);
        return Status.SUCCESS;
    }

    private Status calClusteringResult() {
        // get prediction
        List<MSTensor> outputs = this.model.getOutputsByNodeName("Default/net_with_loss-LossNetwork/_backbone-AutoEncoder/decoder-SequentialCell/0-Dense/BiasAdd-op152");
        if (outputs.isEmpty()) {
            LOGGER.severe("Cannot find outputs tensor for calClusteringResult");
            return Status.FAILED;
        }
        MSTensor out = outputs.get(0);
        Integer predictLabel = getClusteringLabel(out);
        predictions.add(predictLabel);

        // get data of the first user
        if (steps == 0){
            List<MSTensor> inputs = this.model.getInputs();
            Log.d("input size", inputs.get(1).getShape()[0]+" "+inputs.get(1).getShape()[1]);
            MSTensor labelTensor = inputs.get(1);
            float[] user_labels = labelTensor.getFloatData();

            MaxHeap sticky_labels = new MaxHeap(Arrays.copyOf(user_labels,35));
            MaxHeap frequent_labels = new MaxHeap(Arrays.copyOfRange(user_labels, 35, 70));
            MaxHeap longterm_labels = new MaxHeap(Arrays.copyOfRange(user_labels, 70, 105));
            int[] sticky_id = sticky_labels.getTopKIndexes(5);
            int[] frequent_id = frequent_labels.getTopKIndexes(5);
            int[] longterm_id = longterm_labels.getTopKIndexes(3);

            List<Integer> true_labels = new ArrayList<>();
            for(int i=0; i<sticky_id.length; i++){
                true_labels.add(sticky_id[i]);
            }
            for(int i=0; i<frequent_id.length; i++){
                true_labels.add(frequent_id[i]);
            }
            for(int i=0; i<longterm_id.length; i++){
                true_labels.add(longterm_id[i]);
            }

            List<Integer> predicted_labels = new ArrayList<>();
            for(int i=0; i<true_labels.size(); i++){
                predicted_labels.add(groupLabels[predictLabel][i]);
            }
            one_user_labels = true_labels;
            one_predicted_labels = predicted_labels;
        }
        return Status.SUCCESS;
    }
}
