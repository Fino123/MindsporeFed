package com.example.mindsporefederatedlearning.mlp;

import android.annotation.SuppressLint;
import android.net.SSLCertificateSocketFactory;
import android.os.Build;

import androidx.annotation.RequiresApi;

import com.example.mindsporefederatedlearning.common.CommonParameter;
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

public class FlJobMlp {
    private static final Logger LOGGER = Logger.getLogger(FlJobMlp.class.toString());
    private String parentPath;
    private SyncFLJob train_job;
    public FlJobMlp(String parentPath) {
        this.parentPath = parentPath;
    }
    // Android的联邦学习训练任务
    @SuppressLint("NewApi")
    @RequiresApi(api = Build.VERSION_CODES.M)
    public FLClientStatus syncJobTrain() {
        int client_id = CommonParameter.ClientID;
        // 构造dataMap
        String dataPath = this.parentPath + "/data/client"+client_id+"/test-data-int-5000.txt";
        String labelPath = this.parentPath + "/data/client"+client_id+"/label-int-5000.txt";      // 非必须，getModel之后不进行验证可不设置
        String maskPath = this.parentPath + "/data/client"+client_id+"/mask-int-5000.txt";
//        String vocabFile = this.parentPath + "/data/vocab.txt";                // 数据预处理的词典文件路径
//        String idsFile = this.parentPath + "/data/vocab_map_ids.txt";   // 词典的映射id文件路径
        Map<RunType, List<String>> dataMap = new HashMap<>();
        List<String> trainPath = new ArrayList<>();
        trainPath.add(dataPath);
        trainPath.add(labelPath);
        trainPath.add(maskPath);

        String evalDataPath = this.parentPath + "/data/client"+client_id+"test/test-data-int-5000.txt";
        String evalLabelPath = this.parentPath + "/data/client"+client_id+"test/label-int-5000.txt";      // 非必须，getModel之后不进行验证可不设置
        String evalMaskPath = this.parentPath + "/data/client"+client_id+"test/mask-int-5000.txt";
        List<String> evalPath = new ArrayList<>();    // 非必须，getModel之后不进行验证可不设置
        evalPath.add(evalDataPath);                  // 非必须，getModel之后不进行验证可不设置
        evalPath.add(evalLabelPath);                  // 非必须，getModel之后不进行验证可不设置
        evalPath.add(evalMaskPath);

        dataMap.put(RunType.TRAINMODE, trainPath);
        dataMap.put(RunType.EVALMODE, evalPath);      // 非必须，getModel之后不进行验证可不设置

        String flName = "com.example.mindsporefederatedlearning.mlp.MLPClient";                             // AlBertClient.java 包路径
//        String trainModelPath = "/model/albert_inference.mindir.ms";                      // 绝对路径
        String trainModelPath = "/model/MEAN_MLP_hym_train_0104.ms";
        String inferModelPath = "/model/MEAN_MLP_hym_train_0104.ms";                      // 绝对路径, 和trainModelPath保持一致
        String sslProtocol = "TLSv1.2";
        String deployEnv = "android";

        // 端云通信url，请保证Android能够访问到server，否则会出现connection failed
        String domainName = "http://192.168.199.162:9022";
        boolean ifUseElb = true;
//        String domainName = "https://test-kolun-fl.transsion-os.com";
//        boolean ifUseElb = false;
        int serverNum = 1;
        int threadNum = 1;
        BindMode cpuBindMode = BindMode.NOT_BINDING_CORE;
        int batchSize = CommonParameter.batchSize;

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
        flParameter.setSleepTime(50000);


//        List<String> list = new ArrayList<>();
//        list.add(trainTxtPath);
//        List<String> list2 = new ArrayList<>();
//        list2.add(evalTxtPath);
//        flParameter.setHybridWeightName(list, RunType.TRAINMODE);
//        flParameter.setHybridWeightName(list2, RunType.INFERMODE);
        // start FLJob
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

    public void syncJobPredict() {
        // 构造dataMap
        String dataPath = this.parentPath + "/data/app_pred/data_test.txt";
        String labelPath = this.parentPath + "/data/app_pred/label_test.txt";      // 非必须，getModel之后不进行验证可不设置
        String maskPath = this.parentPath + "/data/app_pred/mask_test.txt";
        Map<RunType, List<String>> dataMap = new HashMap<>();
        List<String> inferPath = new ArrayList<>();
        inferPath.add(dataPath);
        inferPath.add(labelPath);
        inferPath.add(maskPath);
        dataMap.put(RunType.INFERMODE, inferPath);

        String flName = "com.example.mindsporefederatedlearning.mlp.MLPClient";                             // AlBertClient.java 包路径         // 绝对路径
        String inferModelPath = "/model/MEAN_MLP_hym_train_1204.ms";                // 绝对路径, 和trainModelPath保持一致;
        int threadNum = 1;
        BindMode cpuBindMode = BindMode.NOT_BINDING_CORE;
        int batchSize = CommonParameter.batchSize;

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
}
