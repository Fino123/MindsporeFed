package com.example.mindsporefederatedlearning.common;

import java.util.Arrays;

/**
 * @author Administrator
 * 类名：MaxHeap
 * 功能：实现大顶堆
 */
public class MaxHeap{
    /** 节点的值 */
    private final float[] values;
    /** 节点的id */
    private final int[] indexes;
    /** 堆的大小 */
    private int heapSize;

    /**
     * 构造函数
     * 功能：根据给定的数组创建一个大顶堆
     * @param arr: 一个数组
     */
    public MaxHeap (float[] arr){
        values = Arrays.copyOf(arr, arr.length);
        indexes = new int[arr.length];
        for (int i=0;i<arr.length;i++) {
            indexes[i] = i;
        }
        buildMaxHeap();
        heapSize = arr.length;
    }

    /**
     * 方法名：maxHeapify()
     * 功能：从某个节点处开始堆化
     * @param position：节点id
     * @param heapSize：堆的大小
     */
    private void maxHeapify(int position, int heapSize) {
        int left = left(position);
        int right = right(position);
        int maxPosition = position;

        if (left < heapSize && values[left] > values[position]) {
            maxPosition = left;
        }

        if (right < heapSize && values[right] > values[maxPosition]) {
            maxPosition = right;
        }

        if (position != maxPosition) {
            //交换值
            swap(values, position, maxPosition);
            //为了返回下标，这里还要交换索引
            swap(indexes, position, maxPosition);
            maxHeapify(maxPosition, heapSize);
        }

    }

    /**
     * 方法名：getTopkIndexes()
     * 功能：获取大顶堆的top-k个节点
     * @param k: top k 值
     * @return results：最大的 k 个元素
     */
    public int[] getTopkIndexes(int k){
        if (k<=getHeapSize()){
            int[] results = new int[k];
            for (int i=0;i<k;i++){
                results[i] = indexes[0];
                values[0] = values[getHeapSize()-1];
                setHeapSize(getHeapSize()-1);
                maxHeapify(0, getHeapSize());
            }
            return results;
        }else {
            throw new RuntimeException("do not have enough values in priority queue");
        }
    }

    /**
     * 方法名：buildMaxHeap()
     * 功能：调整节点，实现大顶堆
     */
    private void buildMaxHeap(){
        int heapSize = values.length;
        int startId = heapSize / 2 - 1;
        for (int i = startId; i >= 0; i--) {
            maxHeapify(i, heapSize);
        }
    }

    /**
     * 方法名：swap()
     * 功能：交换两个节点在数组中的位置
     * @param array：一个数组
     * @param i：第一个节点的id
     * @param j：第二个节点的id
     */
    private void swap(float[] array, int i, int j) {
        float temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    /**
     * 方法名：swap()
     * 功能：交换两个节点在数组中的位置
     * @param array：一个数组
     * @param i：第一个节点的id
     * @param j：第二个节点的id
     */
    private void swap(int[] array, int i, int j) {
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    /** 左子树位置 */
    private int left(int i) {
        return 2 * i + 1;
    }

    /** 右子树位置 */
    private int right(int i) {
        return 2 * i + 2;
    }

    /** 获取堆的大小 */
    public int getHeapSize() {
        return heapSize;
    }

    /** 设置堆的大小 */
    public void setHeapSize(int value) {
        heapSize = value;
    }
}
