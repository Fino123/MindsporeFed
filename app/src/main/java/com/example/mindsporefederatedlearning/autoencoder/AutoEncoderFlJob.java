package com.example.mindsporefederatedlearning.autoencoder;

import android.annotation.SuppressLint;
import android.net.SSLCertificateSocketFactory;
import android.os.Build;

import androidx.annotation.RequiresApi;

import com.example.mindsporefederatedlearning.autoencoder.AutoencoderClient;
import com.mindspore.flclient.BindMode;
import com.mindspore.flclient.FLClientStatus;
import com.mindspore.flclient.FLParameter;
import com.mindspore.flclient.SyncFLJob;
import com.mindspore.flclient.model.RunType;

import java.net.Socket;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import javax.net.ssl.SSLEngine;
import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.X509ExtendedTrustManager;
import javax.net.ssl.X509TrustManager;

/**
 * @author Administrator
 */
public class AutoEncoderFlJob {
    /** 日志 */
    private static final Logger LOGGER = Logger.getLogger(AutoEncoderFlJob.class.toString());
    /** 文件路径 */
    private String parentPath;
    /** 联邦学习任务 */
    private SyncFLJob trainJob;

    /**
     * 构造函数
     * @param parentPath: 文件路径
     */
    public AutoEncoderFlJob(String parentPath) {
        this.parentPath = parentPath;
    }

    /**
     * 方法名： syncJobTrain()
     * 功能：开启 Android 端的联邦学习训练任务
     */
    @SuppressLint("NewApi")
    @RequiresApi(api = Build.VERSION_CODES.M)
    public FLClientStatus syncJobTrain() {
        // 训练数据的文件路径
        List<String> trainTxtPath = new ArrayList<>();
        trainTxtPath.add(this.parentPath + "/data/exps/client_2_train_data.txt");
        trainTxtPath.add(this.parentPath + "/data/exps/client_2_train_label.txt");

        // 验证数据的文件路径
        List<String> evalTxtPath = new ArrayList<>();
        evalTxtPath.add(this.parentPath + "/data/exps/client_2_test_data.txt");
        evalTxtPath.add(this.parentPath + "/data/exps/client_2_test_label.txt");

        // 测试数据的文件路径
        List<String> testTxtPath = new ArrayList<>();
        testTxtPath.add(this.parentPath + "/data/federated_exps/client_1_test_data.txt");
        testTxtPath.add(this.parentPath + "/data/federated_exps/client_1_test_label.txt");

        // 封装训练数据、验证数据、测试数据
        Map<RunType, List<String>> dataMap = new HashMap<>(16);
        dataMap.put(RunType.TRAINMODE, trainTxtPath);
        dataMap.put(RunType.EVALMODE, evalTxtPath);
        dataMap.put(RunType.INFERMODE, testTxtPath);

        // 定义任务名、模型文件的路径等
        String flName = "com.example.mindsporefederatedlearning.autoencoder.AutoencoderClient";
        String trainModelPath = "/model/AutoEncoder_train.ms";
        String inferModelPath = "/model/AutoEncoder_train.ms";
        String sslProtocol = "TLSv1.2";
        String deployEnv = "android";

        // 端云通信url，请保证Android能够访问到server，否则会出现connection failed
        String domainName = "http://192.168.199.162:9023";
        boolean ifUseElb = true;
        int serverNum = 1;
        int threadNum = 1;
        BindMode cpuBindMode = BindMode.NOT_BINDING_CORE;
        int batchSize = 1;

        // 设置联邦学习任务的参数
        FLParameter flParameter = FLParameter.getInstance();
        flParameter.setFlName(flName);
        flParameter.setDataMap(dataMap);
        flParameter.setTrainModelPath(this.parentPath+trainModelPath);
        flParameter.setInferModelPath(this.parentPath+inferModelPath);
        flParameter.setSslProtocol(sslProtocol);
        flParameter.setDeployEnv(deployEnv);
        flParameter.setDomainName(domainName);
        flParameter.setUseElb(ifUseElb);
        flParameter.setServerNum(serverNum);
        flParameter.setThreadNum(threadNum);
        flParameter.setCpuBindMode(cpuBindMode);
        flParameter.setBatchSize(batchSize);
        flParameter.setSleepTime(5000);

        // 设置用户画像模型的kmeans模型文件的路径
        AutoencoderClient.clusteringPath = this.parentPath+"/model/Kmeans_train.ms";

        // 创建客户端的通信socket
        SSLSocketFactory sslSocketFactory = new SSLCertificateSocketFactory(10000);
        flParameter.setSslSocketFactory(sslSocketFactory);
        X509TrustManager x509TrustManager = new X509ExtendedTrustManager() {
            @Override
            public void checkClientTrusted(X509Certificate[] x509Certificates, String s, Socket socket) throws CertificateException {}
            @Override
            public void checkServerTrusted(X509Certificate[] x509Certificates, String s, Socket socket) throws CertificateException {}
            @Override
            public void checkClientTrusted(X509Certificate[] x509Certificates, String s, SSLEngine sslEngine) throws CertificateException {}
            @Override
            public void checkServerTrusted(X509Certificate[] x509Certificates, String s, SSLEngine sslEngine) throws CertificateException {}
            @Override
            public void checkClientTrusted(X509Certificate[] x509Certificates, String s) throws CertificateException {}
            @Override
            public void checkServerTrusted(X509Certificate[] x509Certificates, String s) throws CertificateException {}
            @Override
            public X509Certificate[] getAcceptedIssuers() {
                return new X509Certificate[0];
            }
        };
        flParameter.setX509TrustManager(x509TrustManager);

        // 创建一个任务实例并运行
        trainJob = new SyncFLJob();
        return trainJob.flJobRun();
    }

    /** Android的联邦学习推理任务 */
    public void syncJobPredict() {
        // 封装测试数据
        List<String> testTxtPath = new ArrayList<>();
        testTxtPath.add(this.parentPath + "/data/ms_test_feat_500.txt");
        testTxtPath.add(this.parentPath + "/data/ms_test_label_500.txt");
        Map<RunType, List<String>> dataMap = new HashMap<>(16);
        dataMap.put(RunType.INFERMODE, testTxtPath);

        // 定义任务名、模型文件的路径等
        String flName = "com.example.mindsporefederatedlearning.autoencoder.AutoencoderClient";
        String inferModelPath = "/model/AutoEncoder_train.ms";
        int threadNum = 1;
        BindMode cpuBindMode = BindMode.NOT_BINDING_CORE;
        int batchSize = 1;

        // 设置联邦学习任务的参数
        FLParameter flParameter = FLParameter.getInstance();
        flParameter.setFlName(flName);
        flParameter.setDataMap(dataMap);
        flParameter.setInferModelPath(this.parentPath+inferModelPath);
        flParameter.setThreadNum(threadNum);
        flParameter.setCpuBindMode(cpuBindMode);
        flParameter.setBatchSize(batchSize);

        // 创建一个测试任务并运行
        SyncFLJob syncFlJob = new SyncFLJob();
        List<Object> labels = syncFlJob.modelInfer();
        LOGGER.info("labels = " + Arrays.toString(labels.toArray()));
    }

    /** 结束训练 */
    public void finishJob(){
        trainJob.stopFLJob();
    }
}