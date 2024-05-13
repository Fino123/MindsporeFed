package com.example.mindsporefederatedlearning.other;

/**
 * @author Administrator
 * 类名：APPListItem
 * 功能：存储APP名字及其对应的图标
 */
public class AppListItem {
    private int iconId;
    private String appName;

    public AppListItem(int iconId, String appName) {
        this.iconId = iconId;
        this.appName = appName;
    }

    /** 获取图标序号 */
    public int getIconId() {
        return iconId;
    }

    /** 获取APP名字 */
    public String getAppName() {
        return appName;
    }
}
