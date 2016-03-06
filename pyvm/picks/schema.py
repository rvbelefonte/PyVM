"""
Schema for building databases
"""
# Tables used by non-Spatialite and Spatialite databases
TABLES = {}
TABLES['events'] = """CREATE TABLE 'events' (event TEXT NOT NULL,
    branchid INTEGER DEFAULT 0, subid INTEGER DEFAULT 0, description TEXT,
    PRIMARY KEY (event))"""

TABLES['picks'] = """CREATE TABLE 'picks' (event TEXT,
    srcid INTEGER,
    recid INTEGER, time REAL, error REAL DEFAULT 0.0,
    PRIMARY KEY (event, srcid, recid),
    FOREIGN KEY(event) REFERENCES events(event),
    FOREIGN KEY(srcid) REFERENCES sources(srcid),
    FOREIGN KEY(recid) REFERENCES receivers(recid))"""

# Basic (non-Spatialite) pick database tables and views
BASIC_TABLES = {}
BASIC_TABLES['sources'] = """CREATE TABLE 'sources' (srcid INTEGER,
    srcx FLOAT, srcy FLOAT, srcz FLOAT, PRIMARY KEY (srcid))"""

BASIC_TABLES['receivers'] = """CREATE TABLE 'receivers' (recid INTEGER,
    recx FLOAT, recy FLOAT, recz FLOAT, PRIMARY KEY (recid))"""

BASIC_VIEWS = {}
BASIC_VIEWS['master_picks'] = """CREATE VIEW 'master_picks'
    AS SELECT events.event AS event, branchid, subid,
    sources.srcid AS srcid, srcx, srcy, srcz,
    receivers.recid AS recid, recx, recy, recz,
    0.0 AS offset, time, error FROM
    'picks' INNER JOIN 'events' ON picks.event=events.event
    INNER JOIN 'sources' ON picks.srcid=sources.srcid
    INNER JOIN 'receivers' ON picks.recid=receivers.recid"""


# Views for building VM Tomography input files
VMTOMO_VIEWS = {}

VMTOMO_VIEWS['vmtomo_receivers'] = """CREATE VIEW 'vmtomo_receivers'
    AS SELECT recid, recx, recy, recz FROM 'receivers'"""

VMTOMO_VIEWS['vmtomo_sources'] = """CREATE VIEW 'vmtomo_sources'
    AS SELECT srcid, srcx, srcy, srcz FROM 'sources'"""

VMTOMO_VIEWS['vmtomo_picks'] = """CREATE VIEW 'vmtomo_picks'
    AS SELECT recid, srcid, branchid, subid, offset, time, error FROM
    'master_picks'"""
