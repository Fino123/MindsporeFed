package com.example.mindsporefederatedlearning.copy;

import android.animation.ObjectAnimator;
import android.annotation.SuppressLint;
import android.app.ActivityManager;
import android.content.Context;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.os.Debug;
import android.util.Log;
import android.view.View;
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
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import com.example.mindsporefederatedlearning.AssetCopyer;
import com.example.mindsporefederatedlearning.R;
import com.example.mindsporefederatedlearning.common.TopKAccuracyCallback;
import com.example.mindsporefederatedlearning.mlp.FlJobMlp;
import com.example.mindsporefederatedlearning.other.AppAdapter;
import com.example.mindsporefederatedlearning.other.AppListItem;
import com.example.mindsporefederatedlearning.other.IconDispatcher;
import com.example.mindsporefederatedlearning.utils.JSONUtil;
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

import lecho.lib.hellocharts.formatter.AxisValueFormatter;
import lecho.lib.hellocharts.formatter.SimpleAxisValueFormatter;
import lecho.lib.hellocharts.model.Axis;
import lecho.lib.hellocharts.model.Line;
import lecho.lib.hellocharts.model.LineChartData;
import lecho.lib.hellocharts.model.PointValue;
import lecho.lib.hellocharts.model.ValueShape;
import lecho.lib.hellocharts.view.LineChartView;


@RequiresApi(api = Build.VERSION_CODES.P)
public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    private String parentPath;
    private ListView lvLabels;
    private ListView lvPreds;
    private ImageView imArrow;
    private ImageView imPhone;
    private ImageView imServer;
    private ScrollView svLog;
    private FlJobMlp flJob;
    private ObjectAnimator animator;
    private ObjectAnimator blingAnimator;
    private LineChartView lossLineView;
    private LineChartView accLineView;
    private LineChartData lossLineData;
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
    private Thread logListener;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        getSupportActionBar().hide();
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);
        // 获取该应用程序在Android系统中的磁盘路径
        this.parentPath = this.getExternalFilesDir(null).getAbsolutePath();
        // copy assets目录下面的资源文件到Android系统的磁盘中
        AssetCopyer.copyAllAssets(this.getApplicationContext(), parentPath);

        // 初始化日志
        String logFolderPath = parentPath + "/log";
        File logFolder = new File(logFolderPath);
        if(!logFolder.exists()){
            logFolder.mkdir();
        }
        // copy assets目录下面的资源文件到Android系统的磁盘中
        AssetCopyer.copyAllAssets(this.getApplicationContext(), parentPath);
        LoggerUtil.setLogFilePath(parentPath + "/log/MyLogFile.log");

        // 新建一个线程，启动联邦学习训练与推理任务
        Button start = (Button) findViewById(R.id.bt_start_fl);
        start.setOnClickListener(this);

        JSONUtil.initJSONObject(parentPath + "/data/id2app_500_with_minor.json");

        imArrow = (ImageView) findViewById(R.id.iv_arrow);
        imPhone = (ImageView) findViewById(R.id.im_phone);
        imServer = (ImageView) findViewById(R.id.im_server);

        lossLineView = (LineChartView) findViewById(R.id.loss_line_view);
        accLineView = (LineChartView) findViewById(R.id.acc_line_view);


        clientCondition = (TextView) findViewById(R.id.tv_client_condition);
        serverCondition = (TextView) findViewById(R.id.tv_server_condition);
        arrowCondition = (TextView) findViewById(R.id.tv_animitation);
        netCondition = (TextView) findViewById(R.id.tv_network_condition);
        memoryCondition = (TextView) findViewById(R.id.tv_memory_condition);
        trainingEpoch = (TextView) findViewById(R.id.tv_training_epochs);
        batchSize = (TextView) findViewById(R.id.tv_batch_size);
        learningRate = (TextView) findViewById(R.id.tv_learning_rate);
        tvLog = (TextView) findViewById(R.id.tv_log);

        lvLabels = (ListView) findViewById(R.id.lv_labels);
        lvPreds = (ListView) findViewById(R.id.lv_preds);

        svLog = (ScrollView) findViewById(R.id.sv_log);
        svLog.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
            @Override

            public void onGlobalLayout() {
                svLog.post(new Runnable() {
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

        String TAG = "FLLiteClient";
        logListener = new Thread(new Runnable() {
            @SuppressLint("SetTextI18n")
            @Override
            public void run() {
                Process mLogcatProc = null;
                BufferedReader reader = null;
                Integer epoch = null;
                int launchTimes = 0;
                while (true){
                    try {
                        //获取logcat日志信息
                        mLogcatProc = Runtime.getRuntime().exec(new String[] { "logcat",TAG+":I *:S", "Common:I *:S", "SyncFLJob:I *:S", "LossCallback:I *:S", "GetModel:I *:S", "UpdateModel:I *:S"});
                        reader = new BufferedReader(new InputStreamReader(mLogcatProc.getInputStream()));
                        String line;
                        while ((line = reader.readLine()) != null) {
                            if (line.indexOf("Verify server") > 0) {
                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        animation_start_downloading();
                                        addLogStringToTextView("手机：从服务器下载全局模型");
                                        addLogStringToTextView("手机：验证服务器是否可以连接");
                                    }
                                });
                            } else if (line.indexOf("startFLJob succeed, curIteration")>0) {
                                launchTimes += 1;
                                epoch = Integer.valueOf(line.substring(line.indexOf("curIteration:")+"curIteration: ".length()));
                                final String temp = String.valueOf(epoch);
                                runOnUiThread(new Runnable() {
                                    @SuppressLint("SetTextI18n")
                                    @Override
                                    public void run() {
                                        trainingEpoch.setText("第 "+ temp +" 轮");
                                        switch (NetUtil.getNetWorkState(getApplicationContext())){
                                            case NetUtil.NETWORK_MOBILE:
                                                netCondition.setText("移动网络连接");
                                                break;
                                            case NetUtil.NETWORK_WIFI:
                                                netCondition.setText("WIFI网络连接");
                                                break;
                                            case NetUtil.NETWORK_NONE:
                                                netCondition.setText("无网络连接");
                                                break;
                                        }
                                        // 获取当前应用程序的 PID
                                        int pid = android.os.Process.myPid();
                                        ActivityManager activityManager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
                                        Debug.MemoryInfo[] memoryInfoArray = activityManager.getProcessMemoryInfo(new int[]{pid});
                                        int totalPss = memoryInfoArray[0].getTotalPss();
                                        memoryCondition.setText(totalPss/1024+"MB");
                                        addLogStringToTextView("手机：验证通过");
                                        addLogStringToTextView("手机：启动联邦学习训练，当前轮次：第"+temp+"轮");
                                    }
                                });

                            } else if(line.indexOf("evaluate model after getting model from server")>0){
                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        animation_stop_downloading();
                                        animation_start_evaluate_model();
                                        addLogStringToTextView("手机：下载全局模型成功");
                                        addLogStringToTextView("手机：验证全局模型文件完整性");
                                    }
                                });
                            }else if(line.indexOf("global train epoch")>0){
                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        animation_stop_evaluate_model();
                                        animation_start_training();
                                        addLogStringToTextView("手机：全局模型文件完整！");
                                        addLogStringToTextView("手机：开始利用本地数据训练模型");
                                    }
                                });
                            }else if(line.indexOf("<FLClient> [train] train succeed")>0){
                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        animation_stop_training();
                                        animation_start_uploading();
                                        addLogStringToTextView("手机：本地训练结束");
                                        addLogStringToTextView("手机：上传训练后的模型文件到服务器");
                                    }
                                });
                            }else if(line.indexOf("updateModel success")>0){
                                if (launchTimes>0) {
                                    runOnUiThread(new Runnable() {
                                        @Override
                                        public void run() {
                                            animation_stop_uploading();
                                            animation_start_waiting();
                                            addLogStringToTextView("手机：上传模型文件成功！");
                                            addLogStringToTextView("手机：等待参与联邦的其它客户端上传模型文件");
                                            addLogStringToTextView("服务器：等待所有客户端上传模型文件");
                                        }
                                    });
                                }
                            } else if (line.indexOf("Get model for iteration")>0) {
                                if (launchTimes>0) {
                                    runOnUiThread(new Runnable() {
                                        @Override
                                        public void run() {
                                            animation_stop_waiting();
                                            addLogStringToTextView("服务器：聚合所有模型文件");
                                            addLogStringToTextView("服务器：得到一个新的全局模型");
                                        }
                                    });
                                }
                            } else if(line.indexOf("<FLClient> [evaluate] evaluate acc: ")>0){
                                if (epoch!=null) {
                                    float newX = epoch.floatValue();
                                    float newY = Float.parseFloat(line.substring(line.indexOf("acc:") + 5));
                                    List<PointValue> accPointValues = accLineData.getLines().get(0).getValues();
                                    accPointValues.add(new PointValue(newX, newY));
                                    // shows example
                                    Map<String, List<Integer>> example = TopKAccuracyCallback.getExample();
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
                                    for (int i=0; i<labelArr.size(); i++){
                                        labelsItems.add(new AppListItem(labelArrIds.get(i), JSONUtil.parseAppId(labelArr.get(i))));
                                    }
                                    for (int i=0; i<predArr.size(); i++){
                                        predsItems.add(new AppListItem(predsArrIds.get(i), JSONUtil.parseAppId(predArr.get(i))));
                                    }
                                    runOnUiThread(new Runnable() {
                                        @Override
                                        public void run() {
                                            ArrayAdapter<AppListItem> labelAdapter = new AppAdapter(MainActivity.this, R.layout.app_item, labelsItems);
                                            ArrayAdapter<AppListItem> predAdapter = new AppAdapter(MainActivity.this, R.layout.app_item, predsItems);
                                            lvLabels.setAdapter(labelAdapter);
                                            lvPreds.setAdapter(predAdapter);
                                            addLogStringToTextView("手机：测试精度为："+newY);
                                        }
                                    });
                                }
                            }else if(line.indexOf("average loss:")>0){
                                if (epoch!=null){
                                    List<PointValue> lossPointValues = lossLineData.getLines().get(0).getValues();
                                    float newX = epoch.floatValue();
                                    int from_index = line.indexOf("loss:") + 5;
                                    float newY = Float.parseFloat(line.substring(from_index, from_index+6));
                                    lossPointValues.add(new PointValue(newX, newY));
                                    runOnUiThread(new Runnable() {
                                        @Override
                                        public void run() {
                                            addLogStringToTextView("手机：平均训练损失："+newY);
                                        }
                                    });
                                }
                            }else if(line.indexOf("the GlobalParameter <batchSize> from server:")>0){
                                int batch_size = Integer.parseInt(line.substring(line.indexOf("from server: ")+"from server: ".length()));
                                batchSize.setText(Integer.toString(batch_size));
                            }else if(line.indexOf("[train] lr for client is:")>0){
                                float lr = Float.parseFloat(line.substring(line.indexOf("lr for client is: ") + "lr for client is: ".length()));
                                learningRate.setText(Float.toString(lr));
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        });

        try {
            Process mLogcatProc = Runtime.getRuntime().exec(new String[] { "logcat","-c"});
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    public void addLogStringToTextView(String s){
        if (tvLog!=null && !s.isEmpty()){
            tvLog.setText(tvLog.getText().toString() + s + '\n');
        }
    }

    public void animation_start_downloading(){
        Log.d("animation", "start_downloading");
        imArrow.setImageResource(R.drawable.left);
        float []x = new float[20];
        for(int i=0;i<20;i++)
            x[i]=(200.0f - i*20.0f);
        animator = ObjectAnimator.ofFloat(imArrow, "translationX", x);
        animator.setDuration(1000);
        animator.setRepeatCount(ObjectAnimator.INFINITE);
        animator.start();
        serverCondition.setText("服务器");
        arrowCondition.setText("下载全局模型");
    }

    public void animation_stop_downloading(){
        Log.d("animation", "stop_downloading");
        if(animator!=null){
            animator.cancel();
            imArrow.setTranslationX(0);
        }
        imArrow.setImageResource(R.drawable.no_connection);
        arrowCondition.setText("无连接");
    }

    public void animation_start_evaluate_model(){
        Log.d("animation", "start_evaluate_model");
        clientCondition.setText("验证模型");
        blingAnimator = ObjectAnimator.ofFloat(imPhone, "alpha", 1f, 0f);
        blingAnimator.setDuration(1000);
        blingAnimator.setRepeatCount(ObjectAnimator.INFINITE);
        blingAnimator.setRepeatMode(ObjectAnimator.REVERSE);
        blingAnimator.start();
    }

    public void animation_stop_evaluate_model(){
        Log.d("animation", "stop_evaluate_model");
        clientCondition.setText("手机");
    }

    public void animation_start_training(){
        Log.d("animation", "start_training");
        clientCondition.setText("训练模型");
    }

    public void animation_stop_training(){
        Log.d("animation", "stop_training");
        if (blingAnimator!=null)
            blingAnimator.cancel();
        imPhone.setAlpha(1.0f);
        clientCondition.setText("手机");
        lossLineView.setLineChartData(lossLineData);
        accLineView.setLineChartData(accLineData);
    }

    public void animation_start_uploading(){
        Log.d("animation", "start_uploading");
        arrowCondition.setText("上传模型");
        imArrow.setImageResource(R.drawable.right);
        float []z = new float[20];
        for(int i=0;i<20;i++)
            z[i]=(-200.0f + i*20.0f);

        animator = ObjectAnimator.ofFloat(imArrow, "translationX", z);
        animator.setDuration(800);
        animator.start();
        animator.setRepeatCount(ObjectAnimator.INFINITE);
    }

    public void animation_stop_uploading(){
        Log.d("animation", "stop_uploading");
        if(animator!=null){
            animator.cancel();
            imArrow.setTranslationX(0);
        }
        imArrow.setImageResource(R.drawable.no_connection);
        arrowCondition.setText("无连接");
    }

    public void animation_start_waiting(){
        Log.d("animation", "start_waiting");
        serverCondition.setText("聚合模型");
        blingAnimator = ObjectAnimator.ofFloat(imServer, "alpha", 1f, 0f);
        blingAnimator.setDuration(800);
        blingAnimator.setRepeatCount(ObjectAnimator.INFINITE);
        blingAnimator.setRepeatMode(ObjectAnimator.REVERSE);
        blingAnimator.start();
    }

    public void animation_stop_waiting(){
        Log.d("animation", "stop_waiting");
        if(blingAnimator!=null) {
            blingAnimator.cancel();
            imServer.setAlpha(1.0f);
        }
        serverCondition.setText("服务器");
    }

    public void animation_reset_everything(){
        if (blingAnimator!=null){
            blingAnimator.cancel();
        }
        if (animator!=null){
            animator.cancel();
        }
        imPhone.setAlpha(1.0f);
        clientCondition.setText("手机");
        imArrow.setTranslationX(0.0f);
        arrowCondition.setText("");
        imServer.setAlpha(1.0f);
        serverCondition.setText("服务器");
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()){
            case R.id.bt_start_fl:
                new Thread(() -> {
                    int failed_times = 0;
                    while(true) {
                        flJob = new FlJobMlp(parentPath);
                        FLClientStatus result = flJob.syncJobTrain();
//                        flJob.syncJobPredict();
//                        flJob.finish_job();
                        if (result == FLClientStatus.FAILED) {
                            Log.d("FLClientStatus", "FAILED");
                            failed_times += 1;
                            if (failed_times>5){
                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        Toast.makeText(MainActivity.this, "重启五次失败，结束任务，请重新点击按钮", Toast.LENGTH_SHORT).show();
                                    }
                                });
                                break;
                            }
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    Toast.makeText(MainActivity.this, "训练失败。已经自动重启。", Toast.LENGTH_SHORT).show();
                                    try {
                                        Process mLogcatProc = Runtime.getRuntime().exec(new String[] { "logcat","-c"});
                                    } catch (IOException e) {
                                        throw new RuntimeException(e);
                                    }
                                    animation_reset_everything();
                                }
                            });
                        }else if (result == FLClientStatus.SUCCESS){
                            Log.d("FLClientStatus", "Success");
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    Toast.makeText(MainActivity.this, "训练成功", Toast.LENGTH_SHORT).show();
                                    try {
                                        Process mLogcatProc = Runtime.getRuntime().exec(new String[] { "logcat","-c"});
                                    } catch (IOException e) {
                                        throw new RuntimeException(e);
                                    }
                                    animation_reset_everything();
                                }
                            });
                            break;
                        }
                    }
                }).start();
                logListener.start();
//                String[] allProfiles = {"game", "finance", "video_players", "communication", "social",
//                        "others", "transsion", "maps_and_navigation", "books_and_reference", "tools",
//                        "lifestyle", "dating", "productivity", "personalization", "business", "photography",
//                        "music_and_audio", "shopping", "entertainment", "education", "health_and_fitness",
//                        "travel_and_local", "sports", "news_and_magazines", "food_and_drink", "art_and_design",
//                        "weather", "medical", "parenting", "events", "beauty", "house_and_home", "auto_and_vehicles", "libraries_and_demo", "comics"};
//                List<Integer> labelArr = new ArrayList<>(Arrays.asList(0, 1, 2, 3, 4, 5, 7, 9, 14, 26, 27, 29, 30));
//                List<Integer> predArr = new ArrayList<>(Arrays.asList(0, 1, 2, 3, 4, 5, 8, 16, 25, 26, 27, 28, 29));
//                IconDispatcher dispatcher = new IconDispatcher(IconDispatcher.USER_PROFILE);
//                List<Integer> labelArrIds = dispatcher.transAppIds(labelArr);
//                List<Integer> predsArrIds = dispatcher.transAppIds(predArr);
//                List<AppListItem> labelsItems = new ArrayList<>();
//                List<AppListItem> predsItems = new ArrayList<>();
//                for (int i=0; i<labelArr.size(); i++){
//                    labelsItems.add(new AppListItem(labelArrIds.get(i), allProfiles[i]));
//                }
//                for (int i=0; i<predArr.size(); i++){
//                    predsItems.add(new AppListItem(predsArrIds.get(i), allProfiles[i]));
//                }
//                ArrayAdapter<AppListItem> labelAdapter = new AppAdapter(MainActivity.this, R.layout.app_item, labelsItems);
//                ArrayAdapter<AppListItem> predAdapter = new AppAdapter(MainActivity.this, R.layout.app_item, predsItems);
//                lvLabels.setAdapter(labelAdapter);
//                lvPreds.setAdapter(predAdapter);
                break;

        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
//        flJob.finish_job();
    }

    private void initLineChartData(@NonNull LineChartData data, @NonNull LineChartView lineChartView,
                                   @ColorInt int color, String nameY){
        if (nameY.isEmpty()){
            nameY = "y_default";
        }
        List<PointValue> pointValues = new ArrayList<>();
        List<Line> lines = new ArrayList<>();
        //初始化一条折线
        Line lossLine = new Line(pointValues);
        lossLine.setColor(color);//设置折线颜色
        lossLine.setShape(ValueShape.CIRCLE);//折线图上每个数据点的形状（一共有三种）
        lossLine.setStrokeWidth(2);//设定折线的粗细
        lossLine.setPointRadius(4);//折线图上数据点的半径
        lines.add(lossLine);

        data.setLines(lines);

        //x轴
        Axis axisX = new Axis();
        axisX.setTextSize(10);//x轴字体大小
        axisX.setTextColor(Color.GRAY);//字体颜色
        data.setAxisXBottom(axisX);//设定x轴在底部
        axisX.setName("epochs");
        AxisValueFormatter formatterx = new SimpleAxisValueFormatter(0);
        axisX.setFormatter(formatterx);

        //y轴
        Axis axisY = new Axis();
        axisY.setTextSize(8);//x轴字体大小
        axisY.setTextColor(Color.GRAY);//字体颜色
        data.setAxisYLeft(axisY);//设定y轴在左侧
        axisY.setName(nameY);
        AxisValueFormatter formatter = new SimpleAxisValueFormatter(2);
        axisY.setFormatter(formatter);

        lossLine.setFilled(true);
        lineChartView.setLineChartData(data);
    }
}
