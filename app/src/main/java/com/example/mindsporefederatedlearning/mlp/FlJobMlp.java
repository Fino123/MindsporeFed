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

/**
 * @author Administrator
 */
public class FlJobMlp {
    /** 日志 */
    private static final Logger LOGGER = Logger.getLogger(FlJobMlp.class.toString());
    /** 文件路径 */
    private String parentPath;
    /** 联邦学习任务 */
    private SyncFLJob trainJob;

    /**
     * 构造函数
     * @param parentPath: 文件路径
     */
    public FlJobMlp(String parentPath) {
        this.parentPath = parentPath;
    }

    /** Android的联邦学习训练任务 */
    @SuppressLint("NewApi")
    @RequiresApi(api = Build.VERSION_CODES.M)
    public FLClientStatus syncJobTrain() {
        int clientId = CommonParameter.ClientID;
        // 构造dataMap
        String dataPath = this.parentPath + "/data/client"+clientId+"/test-data-int-5000.txt";
        String labelPath = this.parentPath + "/data/client"+clientId+"/label-int-5000.txt";
        String maskPath = this.parentPath + "/data/client"+clientId+"/mask-int-5000.txt";

        Map<RunType, List<String>> dataMap = new HashMap<>(16);
        List<String> trainPath = new ArrayList<>();
        trainPath.add(dataPath);
        trainPath.add(labelPath);
        trainPath.add(maskPath);

        String evalDataPath = this.parentPath + "/data/client"+clientId+"test/test-data-int-5000.txt";
        // evalLabelPath 非必须，getModel 之后不进行验证可不设置
        String evalLabelPath = this.parentPath + "/data/client"+clientId+"test/label-int-5000.txt";
        String evalMaskPath = this.parentPath + "/data/client"+clientId+"test/mask-int-5000.txt";
        List<String> evalPath = new ArrayList<>();
        evalPath.add(evalDataPath);
        evalPath.add(evalLabelPath);
        evalPath.add(evalMaskPath);

        // 封装训练数据、验证数据
        dataMap.put(RunType.TRAINMODE, trainPath);
        // 非必须，getModel之后不进行验证可不设置
        dataMap.put(RunType.EVALMODE, evalPath);

        // 定义任务名、模型文件的路径等
        String flName = "com.example.mindsporefederatedlearning.mlp.MLPClient";
        String trainModelPath = "/model/MEAN_MLP_hym_train_0104.ms";
        String inferModelPath = "/model/MEAN_MLP_hym_train_0104.ms";
        String sslProtocol = "TLSv1.2";
        String deployEnv = "android";

        // 端云通信url，请保证Android能够访问到server，否则会出现connection failed
        String domainName = "http://192.168.199.162:9022";
        boolean ifUseElb = true;
        int serverNum = 1;
        int threadNum = 1;
        BindMode cpuBindMode = BindMode.NOT_BINDING_CORE;
        int batchSize = CommonParameter.batchSize;

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
        flParameter.setSleepTime(50000);

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
        String dataPath = this.parentPath + "/data/app_pred/data_test.txt";
        // 非必须，getModel之后不进行验证可不设置
        String labelPath = this.parentPath + "/data/app_pred/label_test.txt";
        String maskPath = this.parentPath + "/data/app_pred/mask_test.txt";
        Map<RunType, List<String>> dataMap = new HashMap<>(16);
        List<String> inferPath = new ArrayList<>();
        inferPath.add(dataPath);
        inferPath.add(labelPath);
        inferPath.add(maskPath);
        dataMap.put(RunType.INFERMODE, inferPath);

        // 定义任务名、模型文件的路径等
        String flName = "com.example.mindsporefederatedlearning.mlp.MLPClient";
        String inferModelPath = "/model/MEAN_MLP_hym_train_1204.ms";
        int threadNum = 1;
        BindMode cpuBindMode = BindMode.NOT_BINDING_CORE;
        int batchSize = CommonParameter.batchSize;

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
}
