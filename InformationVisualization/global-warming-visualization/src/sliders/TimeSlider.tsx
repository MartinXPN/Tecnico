import {Slider, Rail, Handles, Tracks, Ticks} from 'react-compound-slider';
import {Handle, Track, Tick} from './components';
import React from "react";
import _ from "lodash";


const sliderStyle: React.CSSProperties = {
    margin: '5%',
    position: 'relative',
    width: '90%'
};

const railStyle: React.CSSProperties = {
    position: 'absolute',
    width: '100%',
    height: 14,
    borderRadius: 7,
    cursor: 'pointer',
    backgroundColor: 'rgb(155,155,155)'
};


interface Props {
    updateValues: (values: number[]) => void;
    initialValues: number[];
    domain: number[];
    colors: string[];
}

export default class TimeSlider extends React.Component<Props> {
    public state = {
        values: this.props.initialValues,
    };

    public onChange = (values: number[]) => {
        this.setState({values});
        if (!_.isEqual(this.state.values, values))
            this.props.updateValues(values);
    };

    public render() {
        const {
            state: {values}
        } = this;

        return (
            <div style={{height: 50, width: '100%'}}>
                <Slider
                    mode={3}
                    step={1}
                    domain={this.props.domain}
                    rootStyle={sliderStyle}
                    // @ts-ignore
                    onUpdate={this.onChange}
                    values={values}>
                    <Rail>
                        {({getRailProps}) => (
                            <div style={railStyle} {...getRailProps()} />
                        )}
                    </Rail>
                    <Handles>
                        {({handles, getHandleProps}) => (
                            <div className="slider-handles">
                                {handles.map(handle => {
                                    return (
                                        <Handle
                                            key={handle.id}
                                            handle={handle}
                                            domain={this.props.domain}
                                            colors={this.props.colors}
                                            getHandleProps={getHandleProps}/>)
                                })}
                            </div>
                        )}
                    </Handles>
                    <Tracks left={false} right={false}>
                        {({tracks, getTrackProps}) => (
                            <div className="slider-tracks">
                                {tracks.map(({id, source, target}) => {
                                        return (
                                            <Track
                                                key={id}
                                                source={source}
                                                target={target}
                                                getTrackProps={getTrackProps}
                                            />)
                                    }
                                )}
                            </div>
                        )}
                    </Tracks>
                    <Ticks count={10}>
                        {({ticks}) => (
                            <div className="slider-ticks">
                                {ticks.map(tick => (
                                    <Tick key={tick.id} tick={tick} count={ticks.length}/>
                                ))}
                            </div>
                        )}
                    </Ticks>
                </Slider>
            </div>
        );
    }
}
