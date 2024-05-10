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

public class AppAdapter extends ArrayAdapter<AppListItem> {
    private List<AppListItem> objects;

    public AppAdapter(@NonNull Context context, int resource, @NonNull List<AppListItem> objects) {
        super(context, resource, objects);
        this.objects = objects;
    }

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
