package com.example.mindsporefederatedlearning.mlp;

import java.util.List;

/**
 * 类名：AppFeature
 * 功能：定义 MLP 模型的输入数据的结构
 * @author Administrator
 */
public class AppFeature {
    /** 数据 */
    List<Integer> data;
    /** 标签 */
    List<Integer> label;
    /** 掩码，用于遮盖数据的某些维度 */
    List<Integer> mask;

    /** 构造函数 */
    public AppFeature(List<Integer> data, List<Integer> label, List<Integer> mask){
        this.data = data;
        this.label = label;
        this.mask = mask;
    }
}
