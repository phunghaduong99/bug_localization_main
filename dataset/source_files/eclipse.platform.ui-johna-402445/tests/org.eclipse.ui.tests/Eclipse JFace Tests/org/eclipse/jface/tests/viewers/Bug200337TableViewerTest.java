/*******************************************************************************
 * Copyright (c) 2007 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 ******************************************************************************/

package org.eclipse.jface.tests.viewers;

import java.util.ArrayList;
import java.util.List;

import org.eclipse.jface.viewers.ArrayContentProvider;
import org.eclipse.jface.viewers.CellEditor;
import org.eclipse.jface.viewers.ICellModifier;
import org.eclipse.jface.viewers.StructuredViewer;
import org.eclipse.jface.viewers.TableViewer;
import org.eclipse.jface.viewers.TextCellEditor;
import org.eclipse.swt.SWT;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.TableColumn;

/**
 * @since 3.3
 *
 */
public class Bug200337TableViewerTest extends ViewerTestCase {

	TableViewer<String,List<String>> tableViewer;

	/**
	 * @param name
	 */
	public Bug200337TableViewerTest(String name) {
		super(name);
	}

	protected StructuredViewer createViewer(Composite parent) {
		tableViewer = new TableViewer<String,List<String>>(parent, SWT.FULL_SELECTION);
		tableViewer.setContentProvider(new ArrayContentProvider<String>(String.class));
		tableViewer.setCellEditors(new CellEditor[] { new TextCellEditor(
				tableViewer.getTable()) });
		tableViewer.setColumnProperties(new String[] { "0" });
		tableViewer.setCellModifier(new ICellModifier() {
			public boolean canModify(Object element, String property) {
				return true;
			}

			public Object getValue(Object element, String property) {
				return "";
			}

			public void modify(Object element, String property, Object value) {
			}

		});

	    new TableColumn(tableViewer.getTable(), SWT.NONE).setWidth(200);

		return tableViewer;
	}

	protected void setUpModel() {
		// don't do anything here - we are not using the normal fModel and
		// fRootElement
	}

	protected void setInput() {

		ArrayList<String> ar = new ArrayList<String>(100);
		for( int i = 0; i < 100; i++ ) {
			ar.add(i, i + "");
		}

		getTableViewer().setInput(ar);
	}

	private TableViewer<String,List<String>> getTableViewer() {
		return tableViewer;
	}

	public void testBug200337() {
		getTableViewer().editElement((String)getTableViewer().getElementAt(0), 0);
		getTableViewer().editElement((String)getTableViewer().getElementAt(90), 0);
	}
}
