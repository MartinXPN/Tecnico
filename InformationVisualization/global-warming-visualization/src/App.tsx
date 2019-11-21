import React, {Component} from 'react';
import './App.css';
import * as d3 from 'd3'
import TimeSlider from "./sliders/TimeSlider";
import SplitPane from 'react-split-pane';
import Map from "./map/Map";
import ScatterPlot from "./charts/ScatterPlot";
import BubbleChart from "./charts/BubbleChart";
import RadialBarChart from "./charts/RadialBarChart";

interface Props {
}

interface State {
    sea2glaciers: d3.DSVParsedArray<{ year: number, level: number, mass: number }> | undefined;
    data: d3.DSVParsedArray<{ country: string, year: number, gdp: number, meat_consumption: number, temperature: number, ghg_emission: number }> | undefined;
    yearStart: number;
    yearEnd: number;
    selectedCountries: Array<string>;
}

export default class App extends Component<Props, State> {

    state = {
        sea2glaciers: undefined,
        data: undefined,
        yearStart: 1970,
        yearEnd: 2014,
        selectedCountries: [],
    };

    componentDidMount(): void {
        d3.json('./sea2glaciers.json').then(data => {
            this.setState({sea2glaciers: data});
            console.log(data);
        });

        d3.json('./gdp2temp2meat2ghg.json').then(data => {
            this.setState({data: data});
            console.log(data);
        });
    }

    render(): React.ReactElement {
        console.log('State:', this.state);
        return (
            <SplitPane split="vertical" minSize='20%' defaultSize='30%' maxSize='50%' allowResize={true}>
                <div style={{width: '100%', height: '100%'}}>
                    <div className="chart-box">
                        {this.state.sea2glaciers &&
                        <RadialBarChart
                            width='100%' height='100%'
                            // @ts-ignore
                            data={this.state.sea2glaciers}
                            yearStart={this.state.yearStart}
                            yearEnd={this.state.yearEnd}
                        />}
                    </div>
                    <div className="chart-box">
                        <BubbleChart width='100%' height='100%' data={this.state.data}/>
                    </div>
                    <div className="chart-box">
                        <ScatterPlot width='100%' height='100%' data={this.state.data}/>
                    </div>
                </div>

                <div style={{width: '100%', height: '100%'}}>
                    <div className="Time-slider">
                        <TimeSlider
                            initialValues={[this.state.yearStart, this.state.yearEnd]}
                            updateValues={(newValues: number[]) => {
                                this.setState({yearStart: newValues[0], yearEnd: newValues[1]})
                            }}/>
                    </div>
                    <Map width='100%' height='100%'/>
                </div>
            </SplitPane>
        );
    }
};
