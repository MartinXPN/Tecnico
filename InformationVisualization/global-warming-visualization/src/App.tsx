import React, {Component} from 'react';
import './App.css';
import * as d3 from 'd3'
import TimeSlider from "./sliders/TimeSlider";
import SplitPane from 'react-split-pane';
import Map from "./map/Map";

interface Props {
}

interface State {
    sea2glaciers: d3.DSVParsedArray<{ year: Date, level: number, mass: number }>;
    data: d3.DSVParsedArray<{ country: string, year: Date, gdp: number, meat_consumption: number, temperature: number, ghg_emission: number }>
    yearStart: number;
    yearEnd: number;
    selectedCountries: Array<string>;
}

export default class App extends Component<Props, State> {

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
        return (
            <SplitPane split="vertical" minSize='20%' defaultSize='30%' maxSize='50%' allowResize={true}>
                <div />
                <div style={{width: '100%', height: '100%'}}>
                    <div className="Time-slider">
                        <TimeSlider />
                    </div>
                    <Map width='100%' height='100%' />
                </div>
            </SplitPane>
        );
    }
};
