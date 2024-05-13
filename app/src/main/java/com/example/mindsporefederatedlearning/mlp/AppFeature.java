package com.example.mindsporefederatedlearning.mlp;

import java.util.List;

public class AppFeature {
    List<Integer> data;
    List<Integer> label;
    List<Integer> mask;

    public AppFeature(List<Integer> data, List<Integer> label, List<Integer> mask){
        this.data = data;
        this.label = label;
        this.mask = mask;
    }
}
