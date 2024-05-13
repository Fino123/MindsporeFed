package com.example.mindsporefederatedlearning.other;

public class AppListItem {
    private int iconId;
    private String appName;

    public AppListItem(int iconId, String appName) {
        this.iconId = iconId;
        this.appName = appName;
    }

    public int getIconId() {
        return iconId;
    }

    public String getAppName() {
        return appName;
    }
}
