package com.example.mindsporefederatedlearning.other;

import com.example.mindsporefederatedlearning.R;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class IconDispatcher {
    private static int[] allIconIds = {
        R.drawable.icon1, R.drawable.icon2, R.drawable.icon3, R.drawable.icon4, R.drawable.icon5, R.drawable.icon6, R.drawable.icon7,
            R.drawable.icon8, R.drawable.icon9, R.drawable.icon10, R.drawable.icon11, R.drawable.icon12, R.drawable.icon13,  R.drawable.icon14,
            R.drawable.icon15, R.drawable.icon16, R.drawable.icon17, R.drawable.icon18, R.drawable.icon19, R.drawable.icon20, R.drawable.icon21,
            R.drawable.icon22, R.drawable.icon23, R.drawable.icon24
    };
    private static int[] profilesIds = {
            R.drawable.game, R.drawable.finance, R.drawable.video_players, R.drawable.communication, R.drawable.social, R.drawable.others,
            R.drawable.transsion, R.drawable.maps_and_navigation, R.drawable.books_and_reference, R.drawable.tools, R.drawable.lifestyle,
            R.drawable.dating, R.drawable.productivity, R.drawable.personalization, R.drawable.business, R.drawable.photography, R.drawable.music_and_audio,
            R.drawable.shopping, R.drawable.entertainment, R.drawable.education, R.drawable.health_and_fitness, R.drawable.travel_and_local, R.drawable.sports,
            R.drawable.news_and_magazines, R.drawable.food_and_drinks, R.drawable.art_and_design, R.drawable.weather, R.drawable.medical, R.drawable.parenting,
            R.drawable.events, R.drawable.beauty, R.drawable.house_and_home, R.drawable.auto_and_vehicles, R.drawable.libraries_and_demo, R.drawable.comics
    };
    public static int APP_RECOMMEND=10222;
    public static int USER_PROFILE=12333;
    private int cnt;
    private HashMap<Integer, Integer> map;
    private int predictMode;
    public IconDispatcher(int predictMode){
        cnt = 0;
        map = new HashMap<>();
        if (predictMode!=APP_RECOMMEND && predictMode!=USER_PROFILE){
            throw new IllegalArgumentException("predictMode can be IconDispatcher.APP_RECOMMEND or IconDispatcher.USER_PROFILE.");
        }
        this.predictMode = predictMode;
    }

    public List<Integer> transAppIds(List<Integer> appIds){
        int[] icons = predictMode==APP_RECOMMEND? allIconIds:profilesIds;
        List<Integer> result = new ArrayList<>();
        for (Integer id:appIds){
            if (map.containsKey(id)){
                result.add(map.get(id));
            }else{
                map.put(id, icons[cnt]);
                result.add(icons[cnt]);
                cnt++;
                if (cnt>=icons.length){
                    throw new IndexOutOfBoundsException("超过ICON数量上届");
                }
            }
        }
        return result;
    }

}
