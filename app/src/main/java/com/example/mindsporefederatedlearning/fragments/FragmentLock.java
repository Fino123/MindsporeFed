package com.example.mindsporefederatedlearning.fragments;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @author Administrator
 * 互斥锁，单例模式，用于保障同时只有一个训练算法启动（App预测或者用户画像）
 */
public class FragmentLock {
    /** 互斥锁实例（单例模式） */
    private static ReentrantLock instance;

    /**
     * 方法名：tryLock
     * 方法功能：在一定等待时间内获取互斥锁
     * @param length 尝试获取锁等待时间的时间长度
     * @param timeUnit 尝试获取锁等待时间的时间单位，如length=1，timeUnit=TimeUnit.SECOND,则表示等待时间为1s
     *
     * @return success: 获取互斥锁是否成功，true为成功，false为失败
     */
    public static boolean tryLock(long length, TimeUnit timeUnit) throws InterruptedException {
        synchronized (FragmentLock.class) {
            if (instance == null) {
                instance = new ReentrantLock(true);
            }
        }
        int len = 2;
        if (timeUnit.toSeconds(length)>len){
            length = 2;
            timeUnit = TimeUnit.SECONDS;
        }
        boolean success = instance.tryLock(length, timeUnit);
        return success;
    }

    /**
     * 方法名：unLock
     * 方法功能：释放锁
     * @return null
     */
    public static void unLock(){
        if (instance==null)
            throw new NullPointerException("instance 还没创建");
        instance.unlock();
    }
}
