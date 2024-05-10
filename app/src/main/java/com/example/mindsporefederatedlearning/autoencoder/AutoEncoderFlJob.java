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


public class AutoEncoderFlJob {
    private static final Logger LOGGER = Logger.getLogger(AutoEncoderFlJob.class.toString());
    private String parentPath;
    private SyncFLJob train_job;
    public AutoEncoderFlJob(String parentPath) {
        this.parentPath = parentPath;
    }
    // Android的联邦学习训练任务
    @SuppressLint("NewApi")
    @RequiresApi(api = Build.VERSION_CODES.M)
    public FLClientStatus syncJobTrain() {
        // 构造dataMap
        List<String> trainTxtPath = new ArrayList<>();
        //trainTxtPath.add(this.parentPath + "/data/federated_exps/client_1_train_data.txt");
        //trainTxtPath.add(this.parentPath + "/data/federated_exps/client_1_train_label.txt");
        trainTxtPath.add(this.parentPath + "/data/exps/client_2_train_data.txt");
        trainTxtPath.add(this.parentPath + "/data/exps/client_2_train_label.txt");

        List<String> evalTxtPath = new ArrayList<>();
        //evalTxtPath.add(this.parentPath + "/data/federated_exps/client_1_test_data.txt");
        //evalTxtPath.add(this.parentPath + "/data/federated_exps/client_1_test_label.txt");
        evalTxtPath.add(this.parentPath + "/data/exps/client_2_test_data.txt");
        evalTxtPath.add(this.parentPath + "/data/exps/client_2_test_label.txt");

        List<String> testTxtPath = new ArrayList<>();
        testTxtPath.add(this.parentPath + "/data/federated_exps/client_1_test_data.txt");
        testTxtPath.add(this.parentPath + "/data/federated_exps/client_1_test_label.txt");

        Map<RunType, List<String>> dataMap = new HashMap<>();
        dataMap.put(RunType.TRAINMODE, trainTxtPath);
        dataMap.put(RunType.EVALMODE, evalTxtPath);
        dataMap.put(RunType.INFERMODE, testTxtPath);

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

        AutoencoderClient.clusteringPath = this.parentPath+"/model/Kmeans_train.ms";

        SSLSocketFactory sslSocketFactory = new SSLCertificateSocketFactory(10000);
        flParameter.setSslSocketFactory(sslSocketFactory);
        X509TrustManager x509TrustManager = new X509ExtendedTrustManager() {
            @Override
            public void checkClientTrusted(X509Certificate[] x509Certificates, String s, Socket socket) throws CertificateException {

            }

            @Override
            public void checkServerTrusted(X509Certificate[] x509Certificates, String s, Socket socket) throws CertificateException {

            }

            @Override
            public void checkClientTrusted(X509Certificate[] x509Certificates, String s, SSLEngine sslEngine) throws CertificateException {

            }

            @Override
            public void checkServerTrusted(X509Certificate[] x509Certificates, String s, SSLEngine sslEngine) throws CertificateException {

            }

            @Override
            public void checkClientTrusted(X509Certificate[] x509Certificates, String s) throws CertificateException {

            }

            @Override
            public void checkServerTrusted(X509Certificate[] x509Certificates, String s) throws CertificateException {

            }

            @Override
            public X509Certificate[] getAcceptedIssuers() {
                return new X509Certificate[0];
            }
        };
        flParameter.setX509TrustManager(x509TrustManager);

        train_job = new SyncFLJob();
        return train_job.flJobRun();
    }
    // Android的联邦学习推理任务
    public void syncJobPredict() {
        // 构造dataMap
        List<String> testTxtPath = new ArrayList<>();
        testTxtPath.add(this.parentPath + "/data/ms_test_feat_500.txt");
        testTxtPath.add(this.parentPath + "/data/ms_test_label_500.txt");
        Map<RunType, List<String>> dataMap = new HashMap<>();
        dataMap.put(RunType.INFERMODE, testTxtPath);

        String flName = "com.example.mindsporefederatedlearning.autoencoder.AutoencoderClient";   // AlBertClient.java 包路径
        String inferModelPath = "/model/AutoEncoder_train.ms";                              // 绝对路径, 和trainModelPath保持一致
        int threadNum = 1;
        BindMode cpuBindMode = BindMode.NOT_BINDING_CORE;
        int batchSize = 1;

        FLParameter flParameter = FLParameter.getInstance();
        flParameter.setFlName(flName);
        flParameter.setDataMap(dataMap);
        flParameter.setInferModelPath(this.parentPath+inferModelPath);
        flParameter.setThreadNum(threadNum);
        flParameter.setCpuBindMode(cpuBindMode);
        flParameter.setBatchSize(batchSize);

        // inference
        SyncFLJob syncFLJob = new SyncFLJob();
        List<Object> labels = syncFLJob.modelInfer();
        LOGGER.info("labels = " + Arrays.toString(labels.toArray()));
    }

    public void finish_job(){
        train_job.stopFLJob();
    }
}