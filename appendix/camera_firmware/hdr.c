void hdr_step()
{
    if (!hdrv_enabled) return;
    if (!lv) return;
    if (!is_movie_mode()) return;
    
    static int odd_frame = 0;
    static int frame;
    static int frame_bracketing;
    
    frame++;
    
    if (recording)
    {
#ifdef MOVREC_STATE // sync by Canon frame number
        frame = MVR_FRAME_NUMBER;
#endif
        frame_bracketing++;
    }
    else
    {
        if (!HALFSHUTTER_PRESSED) odd_frame = (frame / video_mode_fps) % 2;
        frame_bracketing=0;
    }
    
    int iso_low, iso_high;
    hdr_get_iso_range(&iso_low, &iso_high);
    
    
    //we will use the first 200~ frames for bracketing, recovering the response function
    static int i=1, i2=1;
    static int j=0, j2=0;
    int num_set = 3;
    int set_size = 128/2; //2^7 = 128
    if(frame_bracketing<num_set*set_size && recording){ //
        if(j>=num_set && i<128){
            i+=2;
            j=0;
        }else{
            j++;
        }
        int shutter_speed = 8000/i;
        FRAME_ISO=iso_low | (iso_low << 8);
        FRAME_SHUTTER_TIMER=49440/(shutter_speed);
        return;
    }else if(frame_bracketing>=num_set*set_size && frame_bracketing < num_set*set_size*2){
        if(j2>=num_set && i2<128){
            i2+=2;
            j2=0;
        }else{
            j2++;
        }
        int shutter_speed = 8000/i2;
        FRAME_ISO=iso_high | (iso_high << 8);
        FRAME_SHUTTER_TIMER=49440/(shutter_speed);
        return;
    }
    i=1;
    i2=1;
    
    
    //    //hack that similar to the ISO
    //    //just have a few sets of cases - using the max shutters for now.
    //    //special case triggering ;)
    //    //if(iso_high == iso_low){
    //	odd_frame = frame % 3;
    //	//don't touch this... very random hack for 60D
    //	//the timing on the 60D is weird and won't go in order
    //	if(odd_frame==0){
    //		int shutter_speed = 8000;
    //		FRAME_ISO=iso_low | (iso_low << 8);
    //		FRAME_SHUTTER_TIMER=49440/(shutter_speed);
    //	}else if(odd_frame==1){
    //		int shutter_speed = 8000/128;
    //		FRAME_ISO=iso_low | (iso_low << 8);
    //		FRAME_SHUTTER_TIMER=49440/(shutter_speed);
    //	}else{
    //		int shutter_speed = 8000/32;
    //		FRAME_ISO=iso_high | (iso_high << 8);
    //		FRAME_SHUTTER_TIMER=49440/(shutter_speed);
    //	}
    //hack that similar to the ISO, need to set the exposure to 1 stop higher to compensate the for
    //just have a few sets of cases - using the max shutters for now.
    //special case triggering ;)
    //if(iso_high == iso_low){
    odd_frame = frame % 3;
    //don't touch this... very random hack for 60D
    //the timing on the 60D is weird and won't go in order
    if(odd_frame==0){
        int shutter_speed = 8000;
        FRAME_ISO=iso_low | (iso_low << 8);
        FRAME_SHUTTER_TIMER=49440/(shutter_speed);
    }else if(odd_frame==1){
        int shutter_speed = 8000/128;
        FRAME_ISO=iso_low | (iso_low << 8);
        FRAME_SHUTTER_TIMER=49440/(shutter_speed);
    }else{
        int shutter_speed = 8000/16;
        FRAME_ISO=iso_high | (iso_high << 8);
        FRAME_SHUTTER_TIMER=49440/(shutter_speed);
    }
    
    
    //}
    //else{
    //	odd_frame = frame % 2;
    //	int iso = odd_frame ? iso_low : iso_high; // ISO 100-1600
    //	FRAME_ISO = iso | (iso << 8);
    //}
    //FRAME_SHUTTER_TIMER = 8000/16*10; //lower the max by 4 stops
    //~ *(uint8_t*)(lv_struct + 0x54) = iso;
}
