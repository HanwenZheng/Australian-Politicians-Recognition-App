package com.hanwen.politiciansAU;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.widget.ImageView;
import android.widget.TextView;

public class PeopleDetailActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_people_detail);
        int imageId = getIntent().getIntExtra("people_image", 0);
        int imageFactionId = getIntent().getIntExtra("faction_image", 0);
        String desc = getIntent().getStringExtra("people_desc");

        ImageView imageView = (ImageView) findViewById(R.id.iv_people_large);
        ImageView imageFactionView = (ImageView) findViewById(R.id.iv_faction);
        TextView textView = (TextView) findViewById(R.id.tv_people_desc);
        imageView.setImageResource(imageId);
        if (imageId != imageFactionId){
            imageFactionView.setImageResource(imageFactionId);
        }
        textView.setText(desc);
        textView.setMovementMethod(ScrollingMovementMethod.getInstance());
    }
}
