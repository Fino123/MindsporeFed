package com.example.mindsporefederatedlearning.utils;

import android.util.Log;

import java.io.IOException;
import java.util.Enumeration;
import java.util.logging.FileHandler;
import java.util.logging.LogManager;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class LoggerUtil {
    public static void setLogFilePath(String logFilePath) {
        try {
            FileHandler fh = new FileHandler(logFilePath);
            fh.setFormatter(new SimpleFormatter());
            Enumeration<String> s = LogManager.getLogManager().getLoggerNames();
            while (s.hasMoreElements()) {
                String n = s.nextElement();
                Logger logger= LogManager.getLogManager().getLogger(n);
                if(logger!=null)
                    logger.addHandler(fh);
                System.out.println(n);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
