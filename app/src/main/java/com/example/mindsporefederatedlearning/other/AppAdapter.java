package com.example.mindsporefederatedlearning.other;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.example.mindsporefederatedlearning.R;

import java.util.List;

/**
 * @author Administrator
 * 类名：AppAdapter
 * 功能：可视化真实标签和预测标签
 */
public class AppAdapter extends ArrayAdapter<AppListItem> {
    /** 标签列表 */
    private List<AppListItem> objects;

    /**
     * 构造函数
     * @param context：activity 主活动
     * @param resource：控件id
     * @param objects：需要展示的APP列表
     */
    public AppAdapter(@NonNull Context context, int resource, @NonNull List<AppListItem> objects) {
        super(context, resource, objects);
        this.objects = objects;
    }

    /** 获取APP列表视图 */
    @NonNull
    @Override
    public View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
        AppListItem item = objects.get(position);
        ViewHolder holder = null;
        View view = null;
        if (convertView==null){
            view = LayoutInflater.from(getContext()).inflate(R.layout.app_item, parent, false);
            holder = new ViewHolder();
            holder.setIcon(view.findViewById(R.id.im_appitem_icon));
            holder.setName(view.findViewById(R.id.tv_appitem_name));
            view.setTag(holder);
        }else{
            view = convertView;
            holder = (ViewHolder) view.getTag();
        }
        holder.getIcon().setImageResource(item.getIconId());
        holder.getName().setText(item.getAppName());
        return view;
    }

    private class ViewHolder{
        ImageView icon;
        TextView name;

        public ImageView getIcon() {
            return icon;
        }

        public TextView getName() {
            return name;
        }

        public void setIcon(ImageView icon) {
            this.icon = icon;
        }

        public void setName(TextView name) {
            this.name = name;
        }
    }
}
