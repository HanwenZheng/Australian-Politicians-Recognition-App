package com.hanwen.politiciansAU;

import android.content.Context;
import android.content.Intent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;
import java.util.List;

public class PeopleAdapter extends ArrayAdapter<People> {
    public PeopleAdapter(Context context, int resource, List<People> objects) {
        super(context, resource, objects);
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        final People people = getItem(position);
        View onePeopleView = LayoutInflater.from(getContext()).inflate(R.layout.people_item, parent, false);
        ImageView imageView = (ImageView) onePeopleView.findViewById(R.id.iv_people_small);
        TextView textView = (TextView) onePeopleView.findViewById(R.id.tv_people_name);
        imageView.setImageResource(people.getImageId());
        textView.setText(people.getName());

        onePeopleView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getContext(), PeopleDetailActivity.class);
                intent.putExtra("people_image", people.getImageId());
                intent.putExtra("faction_image", people.getImageFactionId());
                intent.putExtra("people_desc", people.getDesc());
                getContext().startActivity(intent);
            }
        });
        return onePeopleView;
    }
}
