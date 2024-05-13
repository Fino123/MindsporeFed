package com.example.mindsporefederatedlearning.autoencoder;

import java.util.List;

public class UserFeature {
    List<Float> data;
    List<Integer> labels;

    public UserFeature(List<Float> data, List<Integer> labels){
        this.data = data;
        this.labels = labels;
    }
}
