package com.hanwen.politiciansAU;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.ListView;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        PeopleAdapter peopleAdapter = new PeopleAdapter(this, R.layout.people_item, People.getAllPeoples());

        ListView listView = (ListView) findViewById(R.id.people_listView);

        listView.setAdapter(peopleAdapter);

        /*ConstraintLayout constraintLayout = findViewById(R.id.main_layout);
        AnimationDrawable animationDrawable = (AnimationDrawable) constraintLayout.getBackground();
        animationDrawable.setEnterFadeDuration(3000);
        animationDrawable.setExitFadeDuration(1500);
        animationDrawable.start();*/
    }


}
