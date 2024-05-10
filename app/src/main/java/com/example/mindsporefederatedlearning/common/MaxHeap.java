package com.example.mindsporefederatedlearning.common;

import java.util.Arrays;

public class MaxHeap{
    private final float[] values;
    private final int[] indexes;
    private int heapSize;

    public MaxHeap (float[] arr){
        values = Arrays.copyOf(arr, arr.length);
        indexes = new int[arr.length];
        for (int i=0;i<arr.length;i++) indexes[i] = i;
        buildMaxHeap();
        heapSize = arr.length;
    }

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

    public int[] getTopKIndexes(int k){
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

    private void buildMaxHeap(){
        int heapSize = values.length;
        for (int i = heapSize / 2 - 1; i >= 0; i--) {
            maxHeapify(i, heapSize);
        }
    }

    private void swap(float[] array, int i, int j) {
        float temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    private void swap(int[] array, int i, int j) {
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    // 左子树位置
    private int left(int i) {
        return 2 * i + 1;
    }

    // 右子树位置
    private int right(int i) {
        return 2 * i + 2;
    }

    public int getHeapSize() {
        return heapSize;
    }

    public void setHeapSize(int value) {
        heapSize = value;
    }
}
