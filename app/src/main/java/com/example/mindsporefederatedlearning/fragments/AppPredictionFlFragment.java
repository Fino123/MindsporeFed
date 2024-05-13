package com.example.mindsporefederatedlearning.fragments;

import android.animation.ObjectAnimator;
import android.annotation.SuppressLint;
import android.app.ActivityManager;
import android.content.Context;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Debug;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.ColorInt;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;

import com.example.mindsporefederatedlearning.AssetCopyer;
import com.example.mindsporefederatedlearning.R;
import com.example.mindsporefederatedlearning.common.TopkAccuracyCallback;
import com.example.mindsporefederatedlearning.mlp.FlJobMlp;
import com.example.mindsporefederatedlearning.other.AppAdapter;
import com.example.mindsporefederatedlearning.other.AppListItem;
import com.example.mindsporefederatedlearning.other.IconDispatcher;
import com.example.mindsporefederatedlearning.utils.JsonUtil;
import com.example.mindsporefederatedlearning.utils.LoggerUtil;
import com.example.mindsporefederatedlearning.utils.NetUtil;
import com.mindspore.flclient.FLClientStatus;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import lecho.lib.hellocharts.formatter.AxisValueFormatter;
import lecho.lib.hellocharts.formatter.SimpleAxisValueFormatter;
import lecho.lib.hellocharts.model.Axis;
import lecho.lib.hellocharts.model.Line;
import lecho.lib.hellocharts.model.LineChartData;
import lecho.lib.hellocharts.model.PointValue;
import lecho.lib.hellocharts.model.ValueShape;
import lecho.lib.hellocharts.view.LineChartView;

/**
 * @author Administrator
 */
public class AppPredictionFlFragment extends Fragment implements View.OnClickListener{
    /** 首次启动时呈现给用户的主活动 */
    private FragmentActivity activity;
    /** 线程池资源 */
    private ThreadPoolExecutor threadPool;
    /** 标记位，用来判断是否可以结束log listener线程*/
    private volatile boolean stopLogListener = false;
    /** 根视图 */
    private View rootView;
    /** 文件路径 */
    private String parentPath;
    /** 真实标签 */
    private ListView lvLabels;
    /** 预测标签 */
    private ListView lvPreds;
    /** 箭头图标 */
    private ImageView imArrow;
    /** 客户端图标 */
    private ImageView imPhone;
    /** 服务器图标 */
    private ImageView imServer;
    /** 日志控件 */
    private ScrollView svLog;
    /** 用户画像算法任务 */
    private FlJobMlp flJob;
    /** 视图动画 */
    private ObjectAnimator animator;
    private ObjectAnimator blingAnimator;
    /** 训练损失视图 */
    private LineChartView lossLineView;
    /** 验证准确率视图 */
    private LineChartView accLineView;
    /** 用于画图的训练损失数据 */
    private LineChartData lossLineData;
    /** 用于画图的准确率数据 */
    private LineChartData accLineData;
    private TextView clientCondition;
    private TextView serverCondition;
    private TextView arrowCondition;
    private TextView tvLog;
    private TextView netCondition;
    private TextView memoryCondition;
    private TextView trainingEpoch;
    private TextView batchSize;
    private TextView learningRate;
    private Runnable logListener;


    /**
     * 构造方法
     * @param executor The ThreadPool resource aims to start a thread for training or log listening.
     */
    public AppPredictionFlFragment(ThreadPoolExecutor executor){
        this.threadPool = executor;
    }

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    /**
     * 方法名：onCreateView()
     * @param inflater The LayoutInflater object that can be used to inflate
     * any views in the fragment,
     * @param container If non-null, this is the parent view that the fragment's
     * UI should be attached to.  The fragment should not add the view itself,
     * but this can be used to generate the LayoutParams of the view.
     * @param savedInstanceState If non-null, this fragment is being re-constructed
     * from a previous saved state as given here.
     *
     * @return rootView: 根页面
     */
    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        rootView = inflater.inflate(R.layout.main_activity_app_pred, container, false);
        return rootView;
    }

    /**
     * 方法名：onViewCreated()
     * @param view The View returned by {@link #onCreateView(LayoutInflater, ViewGroup, Bundle)}.
     * @param savedInstanceState If non-null, this fragment is being re-constructed
     * from a previous saved state as given here.
     */
    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        this.parentPath = activity.getExternalFilesDir(null).getAbsolutePath();
        // copy assets目录下面的资源文件到Android系统的磁盘中
        AssetCopyer.copyAllAssets(activity.getApplicationContext(), parentPath);

        // 初始化日志
        String logFolderPath = parentPath + "/log";
        File logFolder = new File(logFolderPath);
        if(!logFolder.exists()){
            logFolder.mkdir();
        }
        // copy assets目录下面的资源文件到Android系统的磁盘中
        AssetCopyer.copyAllAssets(activity.getApplicationContext(), parentPath);
        LoggerUtil.setLogFilePath(parentPath + "/log/MyLogFile.log");

        // 新建一个线程，启动联邦学习训练与推理任务
        Button start = (Button) rootView.findViewById(R.id.bt_start_fl);
        start.setOnClickListener(this);

        JsonUtil.initJsonObject(parentPath + "/data/id2app_500_with_minor.json");

        imArrow = (ImageView) rootView.findViewById(R.id.iv_arrow);
        imPhone = (ImageView) rootView.findViewById(R.id.im_phone);
        imServer = (ImageView) rootView.findViewById(R.id.im_server);

        lossLineView = (LineChartView) rootView.findViewById(R.id.loss_line_view);
        accLineView = (LineChartView) rootView.findViewById(R.id.acc_line_view);


        clientCondition = (TextView) rootView.findViewById(R.id.tv_client_condition);
        serverCondition = (TextView) rootView.findViewById(R.id.tv_server_condition);
        arrowCondition = (TextView) rootView.findViewById(R.id.tv_animitation);
        netCondition = (TextView) rootView.findViewById(R.id.tv_network_condition);
        memoryCondition = (TextView) rootView.findViewById(R.id.tv_memory_condition);
        trainingEpoch = (TextView) rootView.findViewById(R.id.tv_training_epochs);
        batchSize = (TextView) rootView.findViewById(R.id.tv_batch_size);
        learningRate = (TextView) rootView.findViewById(R.id.tv_learning_rate);
        tvLog = (TextView) rootView.findViewById(R.id.tv_log);

        lvLabels = (ListView) rootView.findViewById(R.id.lv_labels);
        lvPreds = (ListView) rootView.findViewById(R.id.lv_preds);

        svLog = (ScrollView) rootView.findViewById(R.id.sv_log);
        svLog.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
            @Override
            public void onGlobalLayout() {
                svLog.post(new Runnable() {
                    @Override
                    public void run() {
                        svLog.fullScroll(View.FOCUS_DOWN);
                    }
                });
            }
        });

        lossLineData = new LineChartData();
        accLineData = new LineChartData();

        initLineChartData(lossLineData, lossLineView, Color.parseColor("#c6cda1"), "loss");
        initLineChartData(accLineData, accLineView, Color.parseColor("#a1ab6c"), "acc");

        String tag = "FLLiteClient";
        logListener = new Runnable() {
            @SuppressLint("SetTextI18n")
            @Override
            public void run() {
                Process mLogcatProc = null;
                BufferedReader reader = null;
                Integer epoch = null;
                int launchTimes = 0;
                try {
                    //获取logcat日志信息
                    mLogcatProc = Runtime.getRuntime().exec(new String[]{"logcat", tag + ":I *:S", "Common:I *:S", "SyncFLJob:I *:S", "LossCallback:I *:S", "GetModel:I *:S", "UpdateModel:I *:S"});
                    reader = new BufferedReader(new InputStreamReader(mLogcatProc.getInputStream()));
                    String line;
                    while (!stopLogListener) {
                        line = reader.readLine();
                        if(line==null || line.equals(""))
                            continue;
                        if (line.indexOf("Verify server") > 0) {
                            activity.runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    animationStartDownloading();
                                    addLogStringToTextView("手机：从服务器下载全局模型");
                                    addLogStringToTextView("手机：验证服务器是否可以连接");
                                }
                            });
                        } else if (line.indexOf("startFLJob succeed, curIteration") > 0) {
                            launchTimes += 1;
                            epoch = Integer.valueOf(line.substring(line.indexOf("curIteration:") + "curIteration: ".length()));
                            final String temp = String.valueOf(epoch);
                            activity.runOnUiThread(new Runnable() {
                                @SuppressLint("SetTextI18n")
                                @Override
                                public void run() {
                                    trainingEpoch.setText("第 " + temp + " 轮");
                                    switch (NetUtil.getNetWorkState(activity.getApplicationContext())) {
                                        case NetUtil.NETWORK_MOBILE:
                                            netCondition.setText("移动网络连接");
                                            break;
                                        case NetUtil.NETWORK_WIFI:
                                            netCondition.setText("WIFI网络连接");
                                            break;
                                        case NetUtil.NETWORK_NONE:
                                            netCondition.setText("无网络连接");
                                            break;
                                        default:
                                            break;
                                    }
                                    // 获取当前应用程序的 PID
                                    int pid = android.os.Process.myPid();
                                    ActivityManager activityManager = (ActivityManager) activity.getSystemService(Context.ACTIVITY_SERVICE);
                                    Debug.MemoryInfo[] memoryInfoArray = activityManager.getProcessMemoryInfo(new int[]{pid});
                                    int totalPss = memoryInfoArray[0].getTotalPss();
                                    memoryCondition.setText(totalPss / 1024 + "MB");
                                    addLogStringToTextView("手机：验证通过");
                                    addLogStringToTextView("手机：启动联邦学习训练，当前轮次：第" + temp + "轮");
                                }
                            });

                        } else if (line.indexOf("evaluate model after getting model from server") > 0) {
                            activity.runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    animationStopDownloading();
                                    animationStartEvaluateModel();
                                    addLogStringToTextView("手机：下载全局模型成功");
                                    addLogStringToTextView("手机：验证全局模型文件完整性");
                                }
                            });
                        } else if (line.indexOf("global train epoch") > 0) {
                            activity.runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    animationStopEvaluateModel();
                                    animationStartTraining();
                                    addLogStringToTextView("手机：全局模型文件完整！");
                                    addLogStringToTextView("手机：开始利用本地数据训练模型");
                                }
                            });
                        } else if (line.indexOf("<FLClient> [train] train succeed") > 0) {
                            activity.runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    animationStopTraining();
                                    animationStartUploading();
                                    addLogStringToTextView("手机：本地训练结束");
                                    addLogStringToTextView("手机：上传训练后的模型文件到服务器");
                                }
                            });
                        } else if (line.indexOf("updateModel success") > 0) {
                            if (launchTimes > 0) {
                                activity.runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        animationStopUploading();
                                        animationStartWaiting();
                                        addLogStringToTextView("手机：上传模型文件成功！");
                                        addLogStringToTextView("手机：等待参与联邦的其它客户端上传模型文件");
                                        addLogStringToTextView("服务器：等待所有客户端上传模型文件");
                                    }
                                });
                            }
                        } else if (line.indexOf("Get model for iteration") > 0) {
                            if (launchTimes > 0) {
                                activity.runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        animationStopWaiting();
                                        addLogStringToTextView("服务器：聚合所有模型文件");
                                        addLogStringToTextView("服务器：得到一个新的全局模型");
                                    }
                                });
                            }
                        } else if (line.indexOf("<FLClient> [evaluate] evaluate acc: ") > 0) {
                            if (epoch != null) {
                                float newX = epoch.floatValue();
                                float newY = Float.parseFloat(line.substring(line.indexOf("acc:") + 5));
                                List<PointValue> accPointValues = accLineData.getLines().get(0).getValues();
                                accPointValues.add(new PointValue(newX, newY));
                                // shows example
                                Map<String, List<Integer>> example = TopkAccuracyCallback.getExample();
                                List<Integer> labelArr = example.get("label");
                                List<Integer> predArr = example.get("prediction");
                                assert labelArr != null;
                                Collections.sort(labelArr);
                                assert predArr != null;
                                Collections.sort(predArr);
                                IconDispatcher dispatcher = new IconDispatcher(IconDispatcher.APP_RECOMMEND);
                                List<Integer> labelArrIds = dispatcher.transAppIds(labelArr);
                                List<Integer> predsArrIds = dispatcher.transAppIds(predArr);
                                List<AppListItem> labelsItems = new ArrayList<>();
                                List<AppListItem> predsItems = new ArrayList<>();
                                for (int i = 0; i < labelArr.size(); i++) {
                                    labelsItems.add(new AppListItem(labelArrIds.get(i), JsonUtil.parseAppId(labelArr.get(i))));
                                }
                                for (int i = 0; i < predArr.size(); i++) {
                                    predsItems.add(new AppListItem(predsArrIds.get(i), JsonUtil.parseAppId(predArr.get(i))));
                                }
                                activity.runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        ArrayAdapter<AppListItem> labelAdapter = new AppAdapter(activity, R.layout.app_item, labelsItems);
                                        ArrayAdapter<AppListItem> predAdapter = new AppAdapter(activity, R.layout.app_item, predsItems);
                                        lvLabels.setAdapter(labelAdapter);
                                        lvPreds.setAdapter(predAdapter);
                                        addLogStringToTextView("手机：测试精度为：" + newY);
                                    }
                                });
                            }
                        } else if (line.indexOf("average loss:") > 0) {
                            if (epoch != null) {
                                List<PointValue> lossPointValues = lossLineData.getLines().get(0).getValues();
                                float newX = epoch.floatValue();
                                int fromIndex = line.indexOf("loss:") + 5;
                                float newY = Float.parseFloat(line.substring(fromIndex, fromIndex + 6));
                                lossPointValues.add(new PointValue(newX, newY));
                                activity.runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        addLogStringToTextView("手机：平均训练损失：" + newY);
                                    }
                                });
                            }
                        } else if (line.indexOf("the GlobalParameter <batchSize> from server:") > 0) {
                            int batchsize = Integer.parseInt(line.substring(line.indexOf("from server: ") + "from server: ".length()));
                            batchSize.setText(Integer.toString(batchsize));
                        } else if (line.indexOf("[train] lr for client is:") > 0) {
                            float lr = Float.parseFloat(line.substring(line.indexOf("lr for client is: ") + "lr for client is: ".length()));
                            learningRate.setText(Float.toString(lr));
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    try {
                        if (reader != null)
                            reader.close();
                        Runtime.getRuntime().exec(new String[] { "logcat","-c"});
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        };
    }

    /** 在日志框里写入一行新的日志 */
    public void addLogStringToTextView(String s){
        if (tvLog!=null && !s.isEmpty()){
            tvLog.setText(tvLog.getText().toString() + s + '\n');
        }
    }

    /** 从服务器下载全局模型时的页面活动 */
    public void animationStartDownloading(){
        Log.d("animation", "start_downloading");
        imArrow.setImageResource(R.drawable.left);
        int scale = 20;
        float []x = new float[scale];
        for(int i=0;i<scale;i++) {
            x[i]=(200.0f - i*20.0f);
        }
        animator = ObjectAnimator.ofFloat(imArrow, "translationX", x);
        animator.setDuration(1000);
        animator.setRepeatCount(ObjectAnimator.INFINITE);
        animator.start();
        serverCondition.setText("服务器");
        arrowCondition.setText("下载全局模型");
    }

    /** 网络连接失败导致训练停止时的页面活动 */
    public void animationStopDownloading(){
        Log.d("animation", "stop_downloading");
        if(animator!=null){
            animator.cancel();
            imArrow.setTranslationX(0);
        }
        imArrow.setImageResource(R.drawable.no_connection);
        arrowCondition.setText("无连接");
    }

    /** 模型推理时的页面活动 */
    public void animationStartEvaluateModel(){
        Log.d("animation", "start_evaluate_model");
        clientCondition.setText("验证模型");
        blingAnimator = ObjectAnimator.ofFloat(imPhone, "alpha", 1f, 0f);
        blingAnimator.setDuration(1000);
        blingAnimator.setRepeatCount(ObjectAnimator.INFINITE);
        blingAnimator.setRepeatMode(ObjectAnimator.REVERSE);
        blingAnimator.start();
    }

    /** 模型推理结束时的页面活动 */
    public void animationStopEvaluateModel(){
        Log.d("animation", "stop_evaluate_model");
        clientCondition.setText("手机");
    }

    /** 模型训练时的页面活动 */
    public void animationStartTraining(){
        Log.d("animation", "start_training");
        clientCondition.setText("训练模型");
    }

    /** 模型停止训练时的页面活动 */
    public void animationStopTraining(){
        Log.d("animation", "stop_training");
        if (blingAnimator!=null) {
            blingAnimator.cancel();
        }
        imPhone.setAlpha(1.0f);
        clientCondition.setText("手机");
        lossLineView.setLineChartData(lossLineData);
        accLineView.setLineChartData(accLineData);
    }

    /** 客户端上传模型参数时的页面活动 */
    public void animationStartUploading(){
        Log.d("animation", "start_uploading");
        arrowCondition.setText("上传模型");
        imArrow.setImageResource(R.drawable.right);
        int scale = 20;
        float []z = new float[scale];
        for(int i=0;i<scale;i++) {
            z[i]=(-200.0f + i*20.0f);
        }

        animator = ObjectAnimator.ofFloat(imArrow, "translationX", z);
        animator.setDuration(800);
        animator.start();
        animator.setRepeatCount(ObjectAnimator.INFINITE);
    }

    /** 模型上传失败的页面活动 */
    public void animationStopUploading(){
        Log.d("animation", "stop_uploading");
        if(animator!=null){
            animator.cancel();
            imArrow.setTranslationX(0);
        }
        imArrow.setImageResource(R.drawable.no_connection);
        arrowCondition.setText("无连接");
    }

    /** 服务器聚合模型时的页面活动 */
    public void animationStartWaiting(){
        Log.d("animation", "start_waiting");
        serverCondition.setText("聚合模型");
        blingAnimator = ObjectAnimator.ofFloat(imServer, "alpha", 1f, 0f);
        blingAnimator.setDuration(800);
        blingAnimator.setRepeatCount(ObjectAnimator.INFINITE);
        blingAnimator.setRepeatMode(ObjectAnimator.REVERSE);
        blingAnimator.start();
    }

    /** 服务器聚合模型结束时的页面活动 */
    public void animationStopWaiting(){
        Log.d("animation", "stop_waiting");
        if(blingAnimator!=null) {
            blingAnimator.cancel();
            imServer.setAlpha(1.0f);
        }
        serverCondition.setText("服务器");
    }

    /** 重置所有控件 */
    public void animationResetEverything(){
        if (blingAnimator!=null){
            blingAnimator.cancel();
        }
        if (animator!=null){
            animator.cancel();
        }
        imPhone.setAlpha(1.0f);
        clientCondition.setText("手机");
        imArrow.setTranslationX(0.0f);
        imArrow.setImageDrawable(null);
        arrowCondition.setText("");
        imServer.setAlpha(1.0f);
        serverCondition.setText("服务器");
    }

    /** 初始化折线图 */
    @Override
    public void onClick(View view) {
        switch (view.getId()){
            case R.id.bt_start_fl:
                boolean takeLock = false;
                try {
                    takeLock = FragmentLock.tryLock(1, TimeUnit.SECONDS);
                }catch (Exception e){
                    Toast.makeText(activity, "获取互斥锁被打断", Toast.LENGTH_SHORT).show();
                    break;
                }
                if (!takeLock){
                    Toast.makeText(activity, "不可同时启动两个算法", Toast.LENGTH_SHORT).show();
                    break;
                }
                Runnable worker = new Runnable() {
                    @Override
                    public void run() {
                        int failedTimes = 0;
                        flJob = new FlJobMlp(parentPath);
                        try {
                            stopLogListener = false;
                            while(true) {
                                FLClientStatus result = flJob.syncJobTrain();
                                if (result == FLClientStatus.FAILED) {
                                    Log.d("FLClientStatus", "FAILED");
                                    failedTimes += 1;
                                    if (failedTimes>5){
                                        activity.runOnUiThread(()->Toast.makeText(activity, "重启五次失败，结束任务，请重新点击按钮", Toast.LENGTH_SHORT).show());
                                        break;
                                    }
                                    activity.runOnUiThread(()->Toast.makeText(activity, "训练失败。5s后自动自动重启。", Toast.LENGTH_SHORT).show());
                                }else if (result == FLClientStatus.SUCCESS){
                                    Log.d("FLClientStatus", "Success");
                                    activity.runOnUiThread(()->Toast.makeText(activity, "训练成功", Toast.LENGTH_SHORT).show());
                                    break;
                                }
                                activity.runOnUiThread(()->animationResetEverything());
                                Thread.sleep(5000);
                            }
                        }catch (Exception e){
                            e.printStackTrace();
                        } finally {
                            // 释放锁
                            FragmentLock.unLock();
                            // 停下listener
                            stopLogListener = true;
                        }
                    }
                };
                threadPool.execute(worker);
                threadPool.execute(logListener);
                break;
            default:
                break;
        }
    }

    @Override
    public void onAttach(@NonNull Context context) {
        super.onAttach(context);
        this.activity = getActivity();
    }

    /** 初始化折线图 */
    private void initLineChartData(@NonNull LineChartData data, @NonNull LineChartView lineChartView,
                                   @ColorInt int color, String nameY){
        if (nameY.isEmpty()){
            nameY = "y_default";
        }
        List<PointValue> pointValues = new ArrayList<>();
        List<Line> lines = new ArrayList<>();
        //初始化一条折线
        Line lossLine = new Line(pointValues);
        //设置折线颜色
        lossLine.setColor(color);
        //折线图上每个数据点的形状（一共有三种）
        lossLine.setShape(ValueShape.CIRCLE);
        //设定折线的粗细
        lossLine.setStrokeWidth(2);
        //折线图上数据点的半径
        lossLine.setPointRadius(4);
        lines.add(lossLine);

        data.setLines(lines);

        //x轴
        Axis axisX = new Axis();
        //x轴字体大小
        axisX.setTextSize(10);
        //字体颜色
        axisX.setTextColor(Color.GRAY);
        //设定x轴在底部
        data.setAxisXBottom(axisX);
        axisX.setName("epochs");
        AxisValueFormatter formatterx = new SimpleAxisValueFormatter(0);
        axisX.setFormatter(formatterx);

        //y轴
        Axis axisY = new Axis();
        axisY.setTextSize(8);
        axisY.setTextColor(Color.GRAY);
        data.setAxisYLeft(axisY);
        axisY.setName(nameY);
        AxisValueFormatter formatter = new SimpleAxisValueFormatter(2);
        axisY.setFormatter(formatter);

        lossLine.setFilled(true);
        lineChartView.setLineChartData(data);
    }
}
