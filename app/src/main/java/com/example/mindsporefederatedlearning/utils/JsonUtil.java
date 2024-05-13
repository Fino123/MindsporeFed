package com.example.mindsporefederatedlearning.utils;

import com.alibaba.fastjson2.JSONObject;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * @author Administrator
 * 类名：JsonUtil
 * 功能：处理APP相关数据
 */
public class JsonUtil {

    private static JSONObject jsonObject;

    /**
     * 方法名：initJsonObject()
     * 功能：读取json文件数据
     * @param jsonFilePath：json文件路径
     */
    public static void initJsonObject(String jsonFilePath){
        if (jsonObject!=null) {
            return;
        }
        BufferedReader reader = null;
        StringBuilder jsonStrBuilder = new StringBuilder();
        try {
            reader = new BufferedReader(new FileReader(jsonFilePath));
            String line = null;
            while ((line=reader.readLine())!=null) {
                jsonStrBuilder.append(line).append("\n");
            }
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

    /**
     * 方法名：parseAppId()
     * 功能：根据APP的id号解析APP名称
     * @param id：APP的编号
     * @return appName：APP名称
     */
    public static String parseAppId(int id){
        if (jsonObject==null){
            throw new NullPointerException("json object is null");
        }
        String appName = jsonObject.getString(Integer.valueOf(id).toString());
        if (appName==null) {
            throw new NullPointerException("app is not found in json, id is "+id);
        }
        return appName;
    }
}
