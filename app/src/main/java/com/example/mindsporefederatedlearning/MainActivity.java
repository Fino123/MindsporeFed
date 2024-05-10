package com.example.mindsporefederatedlearning;

import android.animation.ObjectAnimator;
import android.annotation.SuppressLint;
import android.app.ActionBar;
import android.app.ActivityManager;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Color;
import android.media.Image;
import android.os.Build;
import android.os.Bundle;
import android.os.Debug;
import android.util.Log;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.ColorInt;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentPagerAdapter;
import androidx.viewpager.widget.ViewPager;

import com.alibaba.fastjson2.JSONObject;
import com.example.mindsporefederatedlearning.common.MaxHeap;
import com.example.mindsporefederatedlearning.common.TopKAccuracyCallback;
import com.example.mindsporefederatedlearning.fragments.AppPredictionFLFragment;
import com.example.mindsporefederatedlearning.fragments.UserProfileFLFragment;
import com.example.mindsporefederatedlearning.mlp.FlJobMlp;
import com.example.mindsporefederatedlearning.other.AppAdapter;
import com.example.mindsporefederatedlearning.other.AppListItem;
import com.example.mindsporefederatedlearning.other.IconDispatcher;
import com.example.mindsporefederatedlearning.utils.JSONUtil;
import com.example.mindsporefederatedlearning.utils.LoggerListener;
import com.example.mindsporefederatedlearning.utils.LoggerUtil;
import com.example.mindsporefederatedlearning.utils.NetUtil;
import com.mindspore.Graph;
import com.mindspore.MSTensor;
import com.mindspore.Model;
import com.mindspore.config.DeviceType;
import com.mindspore.config.MSContext;
import com.mindspore.config.TrainCfg;
import com.mindspore.flclient.FLClientStatus;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import lecho.lib.hellocharts.formatter.AxisValueFormatter;
import lecho.lib.hellocharts.formatter.SimpleAxisValueFormatter;
import lecho.lib.hellocharts.model.Axis;
import lecho.lib.hellocharts.model.AxisValue;
import lecho.lib.hellocharts.model.Line;
import lecho.lib.hellocharts.model.LineChartData;
import lecho.lib.hellocharts.model.PointValue;
import lecho.lib.hellocharts.model.ValueShape;
import lecho.lib.hellocharts.view.LineChartView;


@RequiresApi(api = Build.VERSION_CODES.P)
public class MainActivity extends AppCompatActivity{
    private ViewPager viewPager;
    private Fragment[] fragments = new Fragment[]{new AppPredictionFLFragment(), new UserProfileFLFragment()};
//    private String[] titles = {"APP使用预测", "用户画像预测"};

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        getSupportActionBar().hide();
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);
        viewPager = (ViewPager) findViewById(R.id.view_pager);
        MyPagerAdapter adapter = new MyPagerAdapter(getSupportFragmentManager());
        viewPager.setAdapter(adapter);
    }

    private class MyPagerAdapter extends FragmentPagerAdapter{

        public MyPagerAdapter(FragmentManager manager){
            super(manager, BEHAVIOR_RESUME_ONLY_CURRENT_FRAGMENT);
        }

        @Override
        public int getCount() {
            return fragments.length;
        }

//        @Nullable
//        @Override
//        public CharSequence getPageTitle(int position) {
//            return titles[position];
//        }

        @NonNull
        @Override
        public Fragment getItem(int position) {
            return fragments[position];
        }
    }

}
