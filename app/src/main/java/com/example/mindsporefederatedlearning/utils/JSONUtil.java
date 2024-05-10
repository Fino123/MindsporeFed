package com.example.mindsporefederatedlearning.utils;

import com.alibaba.fastjson2.JSONObject;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.Buffer;

public class JSONUtil {

    private static JSONObject jsonObject;

    public static void initJSONObject(String jsonFilePath){
        if (jsonObject!=null)
            return;
        BufferedReader reader = null;
        StringBuilder jsonStrBuilder = new StringBuilder();
        try {
            reader = new BufferedReader(new FileReader(jsonFilePath));
            String line = null;
            while ((line=reader.readLine())!=null)
                jsonStrBuilder.append(line).append("\n");
        } catch (IOException e){
            e.printStackTrace();
        }finally {
            try {
                if (reader!=null){
                    reader.close();
                }
            }catch (IOException e){
                e.printStackTrace();
            }
        }
        String jsonStr = jsonStrBuilder.toString();
        jsonObject = JSONObject.parseObject(jsonStr);
    }

    public static String parseAppId(int id){
        if (jsonObject==null){
            throw new NullPointerException("json object is null");
        }
        String appName = jsonObject.getString(Integer.valueOf(id).toString());
        if (appName==null)
            throw new NullPointerException("app is not found in json, id is "+id);
        return appName;
    }
}
