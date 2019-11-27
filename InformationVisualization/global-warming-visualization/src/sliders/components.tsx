import React from "react";
import {SliderItem, GetHandleProps, GetTrackProps} from 'react-compound-slider';


export const Handle = ({domain: [min, max], handle: {id, value, percent}, colors, getHandleProps}:
                           { domain: number[], handle: SliderItem, colors: string[], getHandleProps: GetHandleProps }) => {
    const i = +id[id.length - 1];
    const color = colors[i];
    return (
        <div
            role="slider"
            aria-valuemin={min}
            aria-valuemax={max}
            aria-valuenow={value}
            style={{
                left: `${percent}%`,
                position: 'absolute',
                marginLeft: '-11px',
                marginTop: '-6px',
                textAlign: 'center',
                zIndex: 2,
                width: 24,
                height: 24,
                cursor: 'pointer',
                borderRadius: '50%',
                boxShadow: '1px 1px 1px 1px rgba(0, 0, 0, 0.2)',
                backgroundColor: color,
                color: '#333',
            }}
            {...getHandleProps(id)}>

            <div style={{fontSize: 12, marginTop: -25}}>
                {value}
            </div>
        </div>
    );
};


export const Track = ({source, target, getTrackProps}:
                          { source: SliderItem, target: SliderItem, getTrackProps: GetTrackProps }) => (
    <div
        style={{
            position: 'absolute',
            height: 14,
            zIndex: 1,
            backgroundColor: '#7aa0c4',
            borderRadius: 7,
            cursor: 'pointer',
            left: `${source.percent}%`,
            width: `${target.percent - source.percent}%`
        }}
        {...getTrackProps()}
    />
);

export const Tick = ({tick, count}: { tick: SliderItem, count: number }) => (
    <div>
        <div
            style={{
                position: 'absolute',
                marginTop: 14,
                width: 1,
                height: 5,
                backgroundColor: 'rgb(200,200,200)',
                left: `${tick.percent}%`
            }}/>
        <div
            style={{
                position: 'absolute',
                marginTop: 22,
                fontSize: 10,
                textAlign: 'center',
                marginLeft: `${-(100 / count) / 2}%`,
                width: `${100 / count}%`,
                left: `${tick.percent}%`
            }}>
            {tick.value}
        </div>
    </div>
);
