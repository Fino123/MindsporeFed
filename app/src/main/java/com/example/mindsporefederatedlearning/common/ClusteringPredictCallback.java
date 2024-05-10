package com.example.mindsporefederatedlearning.common;

import com.mindspore.MSTensor;
import com.mindspore.Model;
import com.mindspore.flclient.model.Callback;
import com.mindspore.flclient.model.Status;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

public class ClusteringPredictCallback extends Callback {
    private static final Logger LOGGER = Logger.getLogger(ClusteringPredictCallback.class.toString());
    private final List<Integer> predictResults = new ArrayList<>();
    private final int numOfClass;
    private final int batchSize;

    private Model clusteringModel;


    public ClusteringPredictCallback(Model model, Model clustering_model, int batchSize, int numOfClass) {
        super(model);
        this.clusteringModel = clustering_model;
        this.batchSize = batchSize;
        this.numOfClass = numOfClass;
    }

    public List<Integer> getPredictResults() {
        return predictResults;
    }

    @Override
    public Status stepBegin() {
        return Status.SUCCESS;
    }

    @Override
    public Status stepEnd() {
        Status res = calClusteringResult();
        if (res == Status.FAILED) {
            LOGGER.severe("ClusteringPredictCallback stepEnd failed");
            return Status.FAILED;
        }
        return Status.SUCCESS;

    }

    @Override
    public Status epochBegin() {
        return Status.SUCCESS;
    }

    @Override
    public Status epochEnd() {
        LOGGER.info("predictCallback"+predictResults);
        return Status.SUCCESS;
    }

    private Status calClusteringResult() {
        // AutoEncoder
        List<MSTensor> outputs = this.model.getOutputsByNodeName("Default/net_with_loss-LossNetwork/_backbone-AutoEncoder/decoder-SequentialCell/0-Dense/BiasAdd-op152");
        if (outputs.isEmpty()) {
            LOGGER.severe("Cannot find outputs tensor for calClusteringResult");
            return Status.FAILED;
        }
        MSTensor out = outputs.get(0);
        float[] emb = out.getFloatData();
        // Kmeans
        List<MSTensor> clustering_inputs = clusteringModel.getInputs();
        MSTensor data = clustering_inputs.get(0);
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
        predictResults.add(label);
        LOGGER.info("steps:" + steps + ", Clustering label is:" + label);
        return Status.SUCCESS;
    }
}
