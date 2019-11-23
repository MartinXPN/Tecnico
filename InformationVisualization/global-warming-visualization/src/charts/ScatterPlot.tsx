import React, {Component} from "react";
import * as d3 from "d3";
import {GdpTemperatureMeatGhgData} from "../entities";

interface Props {
    width: number | string;
    height: number | string;
    data: d3.DSVParsedArray<GdpTemperatureMeatGhgData>;
    yearStart: number;
    yearEnd: number;
    selectedCountries: Set<string>;
}

interface State {
    countriesDisplayed: Set<string>;
}

export default class ScatterPlot extends Component<Props, State> {
    private readonly countryToData: Map<string, Map<number, GdpTemperatureMeatGhgData>>;
    // @ts-ignore
    private ref: SVGSVGElement;
    // @ts-ignore
    private xScale: d3.ScaleLinear<number, number>;
    // @ts-ignore
    private yScale: d3.ScaleLinear<number, number>;

    state = {
        countriesDisplayed: new Set<string>(),
        countryToData: new Map<string, Map<number, GdpTemperatureMeatGhgData>>(),
    };

    constructor(props: Props) {
        super(props);

        this.countryToData = new Map();
        props.data.forEach(d => {
            if(!this.countryToData.has(d.country))
                this.countryToData.set(d.country, new Map());
            // @ts-ignore
            this.countryToData.get(d.country).set(d.year, d);
        });

        console.log(this.countryToData);
    }

    componentDidMount(): void {
        const svg = d3.select(this.ref);
        const rect = this.ref.getBoundingClientRect();

        const w = rect.width;
        const h = rect.height;
        const padding = rect.width / 10;

        this.xScale = d3.scaleLinear()
            // @ts-ignore
            .domain([d3.min(this.props.data, d => d.ghg_emission), d3.max(this.props.data, d => d.ghg_emission)])
            .range([padding, w - padding * 2]);

        this.yScale = d3.scaleLinear()
            // @ts-ignore
            .domain([d3.min(this.props.data, d => d.temperature), d3.max(this.props.data, d => d.temperature)])
            .range([h - padding, padding]);

        // @ts-ignore
        const xAxis = d3.axisBottom(this.xScale).ticks(5).tickFormat((val: number, _id: number) => {return '' + Math.round(val / 1000000)+ 'M'});
        const yAxis = d3.axisLeft(this.yScale).ticks(5);

        //x axis
        svg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + (h - padding) + ")")
            .call(xAxis);

        //y axis
        svg.append("g")
            .attr("class", "y axis")
            .attr("transform", "translate(" + padding + ", 0)")
            .call(yAxis);
    }

    componentDidUpdate(prevProps: Readonly<Props>, prevState: Readonly<{}>, snapshot?: any): void {
        const h = this.ref.getBoundingClientRect().height;
        const svg = d3.select(this.ref);

        // remove old countries
        this.state.countriesDisplayed.forEach(country => {
            if(!this.props.selectedCountries.has(country)) {
                svg.select(`circle[title='yearStart-${country}']`).remove();
                svg.select(`circle[title='yearEnd-${country}']`).remove();
            }
        });

        // add new countries and display the data
        this.props.selectedCountries.forEach(country => {
            if(!this.state.countriesDisplayed.has(country)) {
                svg.append(`circle`).attr('title', `yearStart-${country}`);
                svg.append(`circle`).attr('title', `yearEnd-${country}`);
            }

            console.log(this.countryToData.get(country));
            console.log(this.props.yearStart);

            // @ts-ignore
            const start = this.countryToData.get(country).get(this.props.yearStart);
            svg.select(`circle[title='yearStart-${country}']`)
                .transition().duration(500)
                .attr('cx', this.xScale(start ? start.ghg_emission: 0))
                .attr('cy', h - this.yScale(start ? start.temperature: 0))
                .attr('r', 4)
                .attr("fill", "green")
                .attr('visibility', start ? 'visible' : 'hidden');


            // @ts-ignore
            const end = this.countryToData.get(country).get(this.props.yearEnd);
            svg.select(`circle[title='yearEnd-${country}']`)
                .transition().duration(500)
                .attr('cx', this.xScale(end ? end.ghg_emission: 0))
                .attr('cy', end ? h - this.yScale(end.temperature) : 0)
                .attr('r', 4)
                .attr("fill", "black")
                .attr('visibility', end ? 'visible' : 'hidden');
        });
    }

    render(): React.ReactElement {
        return (
            <svg className="container"
                 ref={(ref: SVGSVGElement) => this.ref = ref}
                 width={this.props.width}
                 height={this.props.height}>
            </svg>
        );
    }
}
