package com.example.mindsporefederatedlearning.autoencoder;

import java.util.List;

/**
 * 类名：UserFeature
 * 功能：定义 AutoEncoder 模型的输入数据的结构
 * @author Administrator
 */
public class UserFeature {
    /** 用户属性，AutoEncoder 模型的第一个输入 */
    List<Float> data;

    /** 用户标签，AutoEncoder 模型的第二个输入 */
    List<Integer> labels;

    /** 构造函数 */
    public UserFeature(List<Float> data, List<Integer> labels){
        this.data = data;
        this.labels = labels;
    }
}
