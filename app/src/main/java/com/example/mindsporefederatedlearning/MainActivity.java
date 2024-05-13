package com.example.mindsporefederatedlearning;

import android.annotation.SuppressLint;
import android.os.Build;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentPagerAdapter;
import androidx.viewpager.widget.ViewPager;

import com.example.mindsporefederatedlearning.fragments.AppPredictionFlFragment;
import com.example.mindsporefederatedlearning.fragments.UserProfileFlFragment;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;


/**
 * @author Administrator
 * 类名：MainActivity
 * 功能：Android程序的入口
 */
@RequiresApi(api = Build.VERSION_CODES.P)
public class MainActivity extends AppCompatActivity{
    /** 页面视图 */
    private ViewPager viewPager;
    /** 线程池资源 */
    private ThreadPoolExecutor executor;
    /** 工作任务 */
    private Fragment[] fragments;

    /**
     * 覆盖父类接口的 onCreate() 方法
     * 方法名：onCreate(@Nullable Bundle savedInstanceState)
     * 功能：打开APP时呈现主页面
     */
    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        getSupportActionBar().hide();
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);
        viewPager = (ViewPager) findViewById(R.id.view_pager);
        executor = new ThreadPoolExecutor(2, 2, 0, TimeUnit.SECONDS, new LinkedBlockingQueue<>());
        fragments = new Fragment[]{new AppPredictionFlFragment(executor), new UserProfileFlFragment(executor)};
        MyPagerAdapter adapter = new MyPagerAdapter(getSupportFragmentManager());
        viewPager.setAdapter(adapter);
    }

    /**
     * 类名：MyPagerAdapter
     * 功能：指定其中一个算法
     */
    private class MyPagerAdapter extends FragmentPagerAdapter{
        public MyPagerAdapter(FragmentManager manager){
            super(manager, BEHAVIOR_RESUME_ONLY_CURRENT_FRAGMENT);
        }

        /** Android程序集成的算法数量 */
        @Override
        public int getCount() {
            return fragments.length;
        }

        /** 指定某个算法 */
        @NonNull
        @Override
        public Fragment getItem(int position) {
            return fragments[position];
        }
    }

}
