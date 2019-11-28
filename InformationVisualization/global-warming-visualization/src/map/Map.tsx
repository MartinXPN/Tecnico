import React, {Component} from "react";
import * as d3 from 'd3';

interface Props {
    width: number | string;
    height: number | string;
    yearStart: number;
    yearEnd: number;
    hoverCountry: (country: string | undefined) => void;
    hoveredCountry: string | undefined;
    selectedCountries: Set<string>;
    addCountry: (country: string) => void;
    removeCountry: (country: string) => void;
}

interface State {
    currentlyHoveredCountry: string | undefined;
}

export default class Map extends Component<Props, State> {
    private static HOVER_OPACITY = 1;
    private static NORMAL_OPACITY = 0.8;
    private static COLOR = '#3B5988';
    private static HOVER_STROKE = 2;
    private static NORMAL_STROKE = 0.3;

    // @ts-ignore
    private ref: SVGSVGElement;
    // @ts-ignore
    private map: d3.Selection<SVGPathElement, unknown, SVGGElement, unknown>;

    state = {
        currentlyHoveredCountry: undefined,
    };

    drawMap = () => {
        const rect = this.ref.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;

        const projection = d3.geoMercator()
            .scale(150)
            .translate([w / 2, h / 1.5]);
        const path = d3.geoPath().projection(projection);

        // @ts-ignore
        this.map.attr('d', path)
            .style('fill', Map.COLOR)
            .style('stroke', 'white')
            .style('opacity', (d: any) => d.properties.name === this.state.currentlyHoveredCountry ? Map.HOVER_OPACITY : Map.NORMAL_OPACITY)
            .style('stroke-width', (d: any) => d.properties.name === this.state.currentlyHoveredCountry ? Map.HOVER_STROKE : Map.NORMAL_STROKE);

    };


    componentDidUpdate(prevProps: Readonly<Props>, prevState: Readonly<{}>, snapshot?: any): void {
        const map = d3.select(this.ref);

        if (this.state.currentlyHoveredCountry !== this.props.hoveredCountry) {
            if (this.state.currentlyHoveredCountry) {
                map.select(`[title='${this.state.currentlyHoveredCountry}']`)
                    .style('opacity', Map.NORMAL_OPACITY)
                    .style('stroke-width', Map.NORMAL_STROKE);
            }

            if (this.props.hoveredCountry) {
                map.select(`[title='${this.props.hoveredCountry}']`)
                    .style('opacity', Map.HOVER_OPACITY)
                    .style('stroke-width', Map.HOVER_STROKE);
            }

            this.setState({currentlyHoveredCountry: this.props.hoveredCountry});
        }

        /// years were changed => need to render the whole map colors from scratch
        if (this.props.yearStart !== prevProps.yearStart || this.props.yearEnd !== prevProps.yearEnd)
            this.drawMap();
    }

    componentDidMount() {
        d3.json('./world_countries.json').then(data => {
            this.map = d3.select(this.ref)
                .append("g")
                .attr('class', 'map')
                .selectAll("path")
                .data(data.features)
                .enter()
                .append("path").attr('title', (d: any) => d.properties.name)
                .on('click', (d: any) => {
                    const country = d.properties.name;
                    if (this.props.selectedCountries.has(country))
                        this.props.removeCountry(country);
                    else
                        this.props.addCountry(country);
                })
                .on('mouseover', (d: any) => this.props.hoverCountry(d.properties.name))
                .on('mouseout', () => this.props.hoverCountry(undefined));

            this.drawMap();
        });
    }

    render(): React.ReactElement {
        return (
            <svg ref={(ref: SVGSVGElement) => this.ref = ref}
                 width={this.props.width}
                 height={this.props.height}/>
        );
    }
}
