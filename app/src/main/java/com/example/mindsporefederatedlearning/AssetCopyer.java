package com.example.mindsporefederatedlearning;

import android.content.Context;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.logging.Logger;
public class AssetCopyer {
    private static final Logger LOGGER = Logger.getLogger(AssetCopyer.class.toString());
    public static void copyAllAssets(Context context,String destination) {
        LOGGER.info("destination: " + destination);
        copyAssetsToDst(context,"",destination);
    }
    // copy assets目录下面的资源文件到Android系统的磁盘中，具体的路径可打印destination查看
    private static void copyAssetsToDst(Context context,String srcPath, String dstPath) {
        try {
            // 递归获取assets目录的所有的文件名
            String[] fileNames =context.getAssets().list(srcPath);
            if (fileNames.length > 0) {
                // 构建目标file对象
                File file = new File(dstPath);
                //创建目标目录
                file.mkdirs();
                for (String fileName : fileNames) {
                    // copy文件到指定的磁盘
                    if(!srcPath.equals("")) {
                        copyAssetsToDst(context,srcPath + "/" + fileName,dstPath+"/"+fileName);
                    }else{
                        copyAssetsToDst(context, fileName,dstPath+"/"+fileName);
                    }
                }
            } else {
                // 构建源文件的输入流
                InputStream is = context.getAssets().open(srcPath);
                // 构建目标文件的输出流
                FileOutputStream fos = new FileOutputStream(new File(dstPath));
                // 定义1024大小的缓冲数组
                byte[] buffer = new byte[1024];
                int byteCount=0;
                // 源文件写到目标文件
                while((byteCount=is.read(buffer))!=-1) {
                    fos.write(buffer, 0, byteCount);
                }
                // 刷新输出流
                fos.flush();
                // 关闭输入流
                is.close();
                // 关闭输出流
                fos.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
