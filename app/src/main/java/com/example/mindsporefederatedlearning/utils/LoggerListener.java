package com.example.mindsporefederatedlearning.utils;

import android.util.Log;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Stack;

public class LoggerListener {
    private static final String TAG = "FLLiteClient";

    public static void start(){
        new Thread(new Runnable() {
            @Override
            public void run() {
                Process mLogcatProc = null;
                BufferedReader reader = null;
                while (true){
                    try {
                        //获取logcat日志信息
                        mLogcatProc = Runtime.getRuntime().exec(new String[] { "logcat",TAG+":I *:S" });
                        reader = new BufferedReader(new InputStreamReader(mLogcatProc.getInputStream()));
                        String line;

                        while ((line = reader.readLine()) != null) {
                            if (line.indexOf("[startFLJob] startFLJob success") > 0) {
                                Log.d("LogListener", "verify server success");
                            }else if(line.indexOf("evaluate model after getting model from server")>0){
                                Log.d("LogListener", "evaluate model from server");
                            }else if(line.indexOf("global train epoch")>0){
                                Log.d("LogListener", "global train epoch:"+line);
                            }
                        }

                    } catch (Exception e) {

                        e.printStackTrace();

                    }

                }
            }
        }).start();
    }
}
