from py2neo import Graph, Node, Relationship

# Connect to the Neo4j database
graph = Graph("neo4j://localhost:7687", auth=("neo4j", "password"))
#graph.delete_all()


# Create nodes
doctor = Node( "Doctor", node_id=1,name = "Doctor")
patient = Node("Patient", node_id=2 , name = "Patient")
bed = Node("Patient_bed", node_id=3, name = "Patient_bed")
oxygen_mask = Node("Oxygen_mask", node_id=4, name = "Oxygen_mask")
iv_fluids = Node("Iv_fluids", node_id=5, name ="Iv_fluids")
lightings = Node("Lightings", node_id=6, name = "Lightings" )
cardiac_monitor = Node("Cardiac_monitor", node_id=7, name = "Cardiac_monitor")
ventilator = Node("Ventilator", node_id=8, name = "Ventilator")
syringes = Node("Syringe", node_id=9 , name = "Syringe")
stethoscope = Node("Stethoscope", node_id=10, name = "Stethoscope")
icu_room = Node("Icu_room", node_id=11 , name = "Icu_room")
operating_room = Node("Operating_room", node_id=12, name = "Operating_room")
face_mask = Node("Face_mask",node_id= 13,name= "Face_mask")
# Create relationships
treats = Relationship(doctor, "TREATS", patient, relationship_id=1, name ='TREATS')
reads_cardiac_monitor = Relationship(doctor, "READS", cardiac_monitor, relationship_id=2,name ='READS')
checks_ventilator = Relationship(doctor, "CHECKS", ventilator, relationship_id=3 , name ='CHECKS')
uses_stethoscope = Relationship(doctor, "USES", stethoscope, relationship_id=4, name ='USES')
patient_on_bed = Relationship(patient, "LAY_ON", bed, relationship_id=5, name ='LAY_ON')
patient_wears_oxygen_mask = Relationship(patient, "WEARS", oxygen_mask, relationship_id=6, name ='WEARS')
patient_receives_iv_fluids = Relationship(patient, "RECEIVES", iv_fluids, relationship_id=7, name ='RECEIVES')
operating_room_has_lightings = Relationship(operating_room, "HAS", lightings, relationship_id=8, name ='HAS')
operating_room_has_cardiac_monitor = Relationship(operating_room, "HAS", cardiac_monitor, relationship_id=9 ,name ='HAS')
operating_room_has_ventilator = Relationship(operating_room, "HAS", ventilator, relationship_id=10, name ='HAS')
operating_room_has_iv_fluids = Relationship(operating_room, "HAS", iv_fluids, relationship_id=11, name ='HAS')
operating_room_has_bed = Relationship(operating_room, "HAS", bed, relationship_id=12, name ='HAS')
operating_room_has_syringes = Relationship(operating_room, "HAS", syringes, relationship_id=13, name ='HAS')
icu_room_has_bed = Relationship(icu_room, "HAS", bed, relationship_id=14, name ='HAS')
icu_room_has_ventilator = Relationship(icu_room, "HAS", ventilator, relationship_id=15, name ='HAS')
icu_room_has_cardiac_monitor = Relationship(icu_room, "HAS", cardiac_monitor, relationship_id=16, name ='HAS')
icu_room_has_iv_fluids = Relationship(icu_room, "HAS", iv_fluids, relationship_id=17, name ='HAS')
syringes_injectedto_patients = Relationship(syringes,"INJECTED_TO",patient, relationship_id=18, name = "INJECTED_TO")
ivfluids_injectedto_patients = Relationship(iv_fluids,"INJECTED_TO",patient, relationship_id=19, name = "INJECTED_TO")
bed_in_icu = Relationship(bed,"IN",icu_room, relationship_id=20, name = "IN")
bed_in_operating = Relationship(bed,"IN",operating_room, relationship_id=21, name = "IN")
doctor_wears_facemask = Relationship(doctor,"WEARS",face_mask,relationship_id=22, name = "WEARS")

# Add the nodes and relationships to the database
graph.create(doctor)
graph.create(patient)
graph.create(bed)
graph.create( treats)
graph.create(oxygen_mask)
graph.create( iv_fluids)
graph.create( lightings)
graph.create( cardiac_monitor)
graph.create( ventilator)
graph.create( syringes)
graph.create( stethoscope)
graph.create( icu_room)
graph.create( operating_room)
graph.create( treats)
graph.create( reads_cardiac_monitor)
graph.create( checks_ventilator)
graph.create( uses_stethoscope)
graph.create( patient_on_bed)
graph.create( patient_wears_oxygen_mask)
graph.create( patient_receives_iv_fluids)
graph.create( operating_room_has_lightings)
graph.create( operating_room_has_cardiac_monitor)
graph.create( operating_room_has_ventilator)
graph.create( operating_room_has_iv_fluids)
graph.create( operating_room_has_bed)
graph.create( operating_room_has_syringes)
graph.create( icu_room_has_bed)
graph.create( icu_room_has_ventilator)
graph.create( icu_room_has_cardiac_monitor)
graph.create( icu_room_has_iv_fluids)
graph.create(syringes_injectedto_patients)
graph.create(ivfluids_injectedto_patients)
graph.create(bed_in_icu)
graph.create(bed_in_operating)
graph.create(doctor_wears_facemask)
# Query to show all relationships for the label "Patient"
query = "MATCH (p:Patients)-[r]->() RETURN p, r"
results = graph.run(query)
for record in results:
    print(record)
# Run a query to find all nodes labeled "Doctor"
query1 = "MATCH (n:Doctors) RETURN n"
results1 = graph.run(query1).data()
print("Nodes labeled 'Doctor':")
for row in results1:
    print(row['n']['name'])

# Run a query to find all nodes labeled "Thermometer"
''' query2 = "MATCH (n:Node {label: 'patients'}) RETURN n"
results2 = graph.run(query2).data()
print("Nodes labeled 'Thermometer':")
for row in results2:
    print(row['n']) '''

# Run a query to find all relationships involving a node labeled "Doctor" and a node labeled "Thermometer"
query3 = "MATCH (d:Doctors)-[r1]-(n1), (t:Patients)-[r2]-(n2) RETURN d, r1, n1, t, r2, n2"
results3 = graph.run(query3).data()
print("Relationships involving 'Doctor' and 'Thermometer':")
for row in results3:
    print(row['d'], row['r1'], row['n1'], row['t'], row['r2'], row['n2'])


query4 = "MATCH (d:Operating_room)-[r1]-(n1), (t:Ventilator)-[r2]-(n2) RETURN d, r1, n1, t, r2, n2"
results4 = graph.run(query4).data()
print("Relationships involving 'Operating Room' and 'Ventilator':")
for row in results4:
    print(row['d'], row['r1'], row['n1'], row['t'], row['r2'], row['n2'])
    
    
    
import networkx as nx
import matplotlib.pyplot as plt

# Create an empty graph
''' G = nx.Graph()

# Add nodes to the graph
nodes = [doctor, thermometer, patient, bed, oxygen_mask, iv_fluids, lightings, cardiac_monitor, ventilator, syringes, stethoscope, icu_room, operating_room]
G.add_nodes_from(nodes)

# Add edges to the graph
edges = [treats, reads_cardiac_monitor, checks_ventilator, uses_stethoscope, patient_on_bed, patient_wears_oxygen_mask, patient_receives_iv_fluids, operating_room_has_lightings, operating_room_has_cardiac_monitor, operating_room_has_ventilator, operating_room_has_iv_fluids, operating_room_has_bed, operating_room_has_syringes, icu_room_has_bed, icu_room_has_ventilator, icu_room_has_cardiac_monitor, icu_room_has_iv_fluids]
G.add_edges_from([(r.start_node, r.end_node, {'type': r.type}) for r in edges])

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
labels = nx.get_edge_attributes(G,'type')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show() '''
