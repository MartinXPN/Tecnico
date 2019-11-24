import React, {Component} from 'react';
import './App.css';
import * as d3 from 'd3'
import TimeSlider from "./sliders/TimeSlider";
import SplitPane from 'react-split-pane';
import Map from "./map/Map";
import ScatterPlot from "./charts/ScatterPlot";
import BubbleChart from "./charts/BubbleChart";
import RadialBarChart from "./charts/RadialBarChart";
import {SeaGlaciersData} from "./entities";
import {GdpTemperatureMeatGhgData} from "./entities";


interface Props {
}

interface State {
    sea2glaciers: d3.DSVParsedArray<SeaGlaciersData> | undefined;
    data: d3.DSVParsedArray<GdpTemperatureMeatGhgData> | undefined;
    yearStart: number;
    yearEnd: number;
    selectedCountries: Set<string>;
    hoveredCountry: string | undefined;
}

export default class App extends Component<Props, State> {

    state = {
        sea2glaciers: undefined,
        data: undefined,
        yearStart: 1970,
        yearEnd: 2014,
        selectedCountries: new Set(['Armenia', 'Portugal', 'United States']),
        hoveredCountry: undefined,
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

    updateHoveredCountry = (country: string | undefined) => {this.setState({hoveredCountry: country})};
    selectCountry = (country: string) => {this.setState({selectedCountries: new Set([country])})};
    addCountry = (country: string) => {this.setState({selectedCountries: new Set([...Array.from(this.state.selectedCountries), country])})};
    removeCountry = (country: string) => {
        const selectedCountries = new Set(Array.from(this.state.selectedCountries));
        selectedCountries.delete(country);
        this.setState({selectedCountries: selectedCountries});
    };


    render(): React.ReactElement {
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
                        {this.state.data &&
                        <BubbleChart
                            width='100%' height='100%'
                            // @ts-ignore
                            data={this.state.data}
                            yearStart={this.state.yearStart}
                            yearEnd={this.state.yearEnd}
                            selectedCountries={this.state.selectedCountries}
                            selectCountry={this.selectCountry}
                            hoverCountry={this.updateHoveredCountry}
                            currentHoveredCountry={this.state.hoveredCountry}
                        />}
                    </div>
                    <div className="chart-box">
                        {this.state.data &&
                        <ScatterPlot
                            width='100%' height='100%'
                            // @ts-ignore
                            data={this.state.data}
                            yearStart={this.state.yearStart}
                            yearEnd={this.state.yearEnd}
                            selectedCountries={this.state.selectedCountries}
                            selectCountry={this.selectCountry}
                            hoverCountry={this.updateHoveredCountry}
                            currentHoveredCountry={this.state.hoveredCountry}
                        />}
                    </div>
                </div>

                <div style={{width: '100%', height: '100%'}}>
                    <div className="Time-slider">
                        <TimeSlider
                            domain={[1950, 2014]}
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
