create table Event (
    event_id int AUTO_INCREMENT primary key, 
    event_type varchar(255),
    start_time DOUBLE, 
    end_time DOUBLE
);

create table VisibleAt (
    -- video_id int,
    filename varchar(255),
    frame_id int,
    event_id int, 
    x1 double,
    x2 double,
    y1 double,
    y2 double,
    primary key (filename, frame_id, event_id),
    foreign key (event_id) references Event(event_id)
);

create table Car (
    event_id int primary key,
    color varchar(20),
    size double,
    foreign key (event_id) references Event(event_id)
);

create table Person (
    event_id int primary key,
    female boolean,
    upwhite varchar(20),
    foreign key (event_id) references Event(event_id)
);

create table UnderlyingEvents (
    pid int, 
    cid int, 
    foreign key (pid) references Event(event_id), 
    foreign key (cid) references Event(event_id)
);